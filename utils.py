# --------------------------------------------------------------------------------------
# Paths for local RAFT-Stereo and PSMNet modules (adjust if your tree is different)
# --------------------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath("RAFT-Stereo"))
sys.path.append(os.path.abspath("PSMNet"))
# --------------------------------------------------------------------------------------
import cv2
from PIL import Image
import numpy as np
import torch
import open3d as o3d

# RAFT-Stereo utilities / model
from core.utils.utils import InputPadder
from core.raft_stereo import RAFTStereo
# PSMNet model
from PSMNet.models import stackhourglass
import torch.nn.functional as F

# ======================================================================================
# Model loader
# ======================================================================================
def load_model(model_name, model_path):
    """
    Load a stereo model (RAFT-Stereo or PSMNet) on CUDA (if available) and return it.

    Args:
        model_name (str): 'RAFT-Stereo' or 'PSMNet'.
        model_path (str): Path to the pretrained weights.

    Returns:
        torch.nn.Module: The loaded model in eval mode, wrapped by DataParallel.

    Notes:
        - Keeps original structure of your code; prints device and basic logs.
        - Uses strict=False for RAFT-Stereo to tolerate minor key mismatches.
        - Expects PSMNet checkpoints saved with a 'state_dict' key.
    """
        
    # Set Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if model_name == 'RAFT-Stereo':
        # Minimal arg container for RAFT-Stereo init (same as your original)
        class Args:
            def __init__(self):
                self.shared_backbone = True
                self.corr_implementation = 'reg'
                self.corr_levels = 4
                self.corr_radius = 4
                self.n_downsample = 3
                self.context_norm = 'batch'
                self.slow_fast_gru = True
                self.n_gru_layers = 2
                self.hidden_dims = [128]*3
                self.mixed_precision = False
                self.small = False
        # Initialize model
        args = Args()
        model = RAFTStereo(args)
        print(model.update_block.mask[2].weight.shape)

        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.eval()

        # Load pretrained weights
        checkpoint = torch.load(model_path, weights_only=True)

        # Handle the case where checkpoint keys don't have 'module.' prefix
        if not any(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {f'module.{k}': v for k, v in checkpoint.items()}

        # Load state dict with strict=False to handle missing keys
        model.load_state_dict(checkpoint, strict=False)
        print("Model loaded successfully!")
    
    if model_name == 'PSMNet':
        MAX_DISP = 192
        print("=> Loading PSMNet model...")
        model = stackhourglass(maxdisp=MAX_DISP)
        model = torch.nn.DataParallel(model).to(device)

        # Load PSMNet weights (expects 'state_dict')
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict['state_dict'])
        model.eval()
        print("=> Model loaded successfully.")
    
    return model

# ======================================================================================
# Disparity: unified for image paths OR ndarray frames (video)
# ======================================================================================
def compute_disparity(
        model_name, 
        model, 
        left, right, # str paths OR (H,W,3) BGR ndarrays
        left_map1, left_map2, right_map1, right_map2):
    
    """
    Compute disparity from either file paths or BGR ndarrays.
    The function internally rectifies/undistorts using the provided maps.

    Args:
        model_name (str): 'RAFT-Stereo' or 'PSMNet'.
        model (nn.Module): Loaded stereo model.
        left, right (str or np.ndarray): File paths or raw BGR frames (H,W,3).
        left_map1, left_map2, right_map1, right_map2: Rectify/undistort maps.

    Returns:
        np.ndarray: Disparity map (H,W), dtype float (as in original code).

    Notes:
        - Keeps your original tensor preprocessing and padding behavior.
        - For RAFT-Stereo, output list's last element is used (as before).
        - For PSMNet, 16-alignment padding then unpadding is applied.
        - Tensors are moved to GPU by calling .cuda() as in your code.
          (This keeps behavior as-is; consider assigning back to variables
           if you want to be explicit about device placement.)
    """
    # Load or accept frames
    if isinstance(left, str):
        left_img = cv2.imread(left,  cv2.IMREAD_COLOR)
        assert left_img is not None, f"读不到左图: {left}"
    else:
        left_img = left
    if isinstance(right, str):
        right_img = cv2.imread(right, cv2.IMREAD_COLOR)
        assert right_img is not None, f"读不到右图: {right}"
    else:
        right_img = right

    # Rectify + undistort
    left_rect = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LANCZOS4)
    right_rect = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LANCZOS4)

    # To NCHW float [0,1]
    left_rect_torch = torch.from_numpy(left_rect).permute(2, 0, 1).float()[None] / 255.0
    right_rect_torch = torch.from_numpy(right_rect).permute(2, 0, 1).float()[None] / 255.0

    # Keep original behavior: move to CUDA (no reassignment)
    left_rect_torch.cuda()
    right_rect_torch.cuda()

    if model_name == 'RAFT-Stereo':
        # RAFT-Stereo: pad -> infer -> unpad
        padder = InputPadder(left_rect_torch.shape)
        left_rect_torch, right_rect_torch = padder.pad(left_rect_torch, right_rect_torch)
        with torch.no_grad():
            disp = model(left_rect_torch, right_rect_torch)
            disp = padder.unpad(disp[-1]) # use last stage output
        disp = disp.squeeze().detach().cpu().numpy()

    if model_name == 'PSMNet':
        # PSMNet: 16-alignment padding
        _, _, h, w = left_rect_torch.shape
        top_pad = 16 - (h % 16) if h % 16 != 0 else 0
        right_pad = 16 - (w % 16) if w % 16 != 0 else 0

        left_rect_torch = F.pad(left_rect_torch, (0, right_pad, top_pad, 0))
        right_rect_torch = F.pad(right_rect_torch, (0, right_pad, top_pad, 0))
        with torch.no_grad():
            disp = model(left_rect_torch, right_rect_torch) 
        disp = disp.squeeze().detach().cpu().numpy()
        disp = disp[top_pad:, :w] # remove padding

    return disp

# ======================================================================================
# Point cloud generation (path OR ndarray for the left image)
# ======================================================================================
def generate_pointcloud(
        model_name, 
        disp, 
        left, 
        left_map1, left_map2, 
        Q):
    """
    Generate an Open3D point cloud from a disparity map and left image.
    Accepts either a left image path or a raw left BGR ndarray. The left image
    is rectified/undistorted using the provided maps, and colors are sampled
    from the rectified left image.

    Args:
        model_name (str): 'RAFT-Stereo' or 'PSMNet'. (Determines disparity sign)
        disp (np.ndarray): Disparity (H,W).
        left (str or np.ndarray): Left image path or BGR ndarray (H,W,3).
        left_map1, left_map2: Rectify/undistort maps for the left image.
        Q (np.ndarray): Reprojection matrix from stereoRectify.

    Returns:
        open3d.geometry.PointCloud: Colored point cloud.

    Notes:
        - Keeps your original Y/Z sign flips for Open3D viewing:
              points[:,1] *= -1
              points[:,2] *= -1
          This is convenient for certain visual conventions in Open3D,
          but is not required for reprojecting back to original images.
        - Black-ish pixels are filtered out by a threshold (as in your code).
    """    

    # Reproject disparity to 3D (note the sign convention kept as-is)
    if model_name == 'RAFT-Stereo':
        points_3D = cv2.reprojectImageTo3D(-disp, Q)
        mask_disp = -disp > 0
    if model_name == 'PSMNet':
        points_3D = cv2.reprojectImageTo3D(disp, Q)
        mask_disp = disp > 0  # adjust if your valid range differs

    # Load left image (path OR ndarray)
    if isinstance(left, str):
        color_src = cv2.imread(left,  cv2.IMREAD_COLOR)
        assert left is not None, f"Cannot Read Left Image: {left}"
    else:
        color_src = left

    # Rectify left image for coloring
    color_src_rect = cv2.remap(color_src, left_map1, left_map2, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    img = cv2.cvtColor(color_src_rect, cv2.COLOR_BGR2RGB)
    # Mask out near-black regions (same threshold as your code)
    black_threshold = 30 
    mask_not_black = np.any(img > black_threshold, axis=2)
    final_mask = mask_disp & mask_not_black

    # Gather points and apply Open3D-friendly sign flips (kept as-is)  
    points = points_3D[final_mask]
    points[:, 1] *= -1 
    points[:, 2] *= -1 

    # Colors from rectified left image (RGB normalized to [0,1] later)
    colors = img[final_mask]

    # Build Open3D point cloud (unchanged)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 归一化到 [0, 1]

    return pcd

# ======================================================================================
# Project/transform point cloud and compute uL/uR on original images
# ======================================================================================
def pcd_to_rectified_images(pcd, stereo_params, image_size):
    """
    将点云(整左坐标)投影到 rectified-left / rectified-right 平面并生成点图。
    Args:
        pcd : Nx3 array or open3d.PointCloud (rectified-left coordinates)
        stereo_params : 包含 'R','T','R_L','R_R','P_L','P_R'
        image_size : (W, H)
    Returns:
        img_rect_L, img_rect_R  (黑底彩色点层)
    """
    if hasattr(pcd, "points"):
        pts_rectL = np.asarray(pcd.points, dtype=np.float64)
    else:
        pts_rectL = np.asarray(pcd, dtype=np.float64)
    assert pts_rectL.ndim == 2 and pts_rectL.shape[1] == 3

    # pts_rectL[:,0]*=-1
    W, H = image_size

    # === rectified-left 投影 ===
    P_L = stereo_params['P_L'].astype(np.float64)
    KpL = P_L[:3, :3]
    uL_rect, _ = cv2.projectPoints(
        pts_rectL.reshape(-1, 1, 3),
        np.zeros((3,1)), np.zeros((3,1)),
        KpL, np.zeros(5)
    )
    uL_rect = uL_rect.reshape(-1, 2)

    # === rectified-right 投影 ===
    R_L = stereo_params['R_L'].astype(np.float64)
    R_R = stereo_params['R_R'].astype(np.float64)
    R   = stereo_params['R'].astype(np.float64)
    T   = stereo_params['T'].astype(np.float64).reshape(3,1)

    R_rect = R_R @ R @ R_L.T
    T_rect = (R_R @ T).reshape(1,3)

    X_R_rect = (R_rect @ pts_rectL.T).T + T_rect

    P_R = stereo_params['P_R'].astype(np.float64)
    KpR = P_R[:3, :3]
    uR_rect, _ = cv2.projectPoints(
        X_R_rect.reshape(-1,1,3),
        np.zeros((3,1)), np.zeros((3,1)),
        KpR, np.zeros(5)
    )
    uR_rect = uR_rect.reshape(-1,2)

    # === 绘制点图 ===
    img_rect_L = np.zeros((H, W, 3), np.uint8)
    img_rect_R = np.zeros((H, W, 3), np.uint8)

    uvL = np.rint(uL_rect).astype(int)
    uvR = np.rint(uR_rect).astype(int)
    for x, y in uvL:
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img_rect_L, (x, y), 1, (0, 0, 255), -1)
    for x, y in uvR:
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img_rect_R, (x, y), 1, (0, 0, 255), -1)

    return img_rect_L, img_rect_R

def inverse_remap_rectified_to_original(rectified_img, stereo_params, camera="left"):
    """
    将整流图像反整流回原始图像平面
    """
    H, W = rectified_img.shape[:2]
    if camera == "left":
        P = stereo_params["P_L"].astype(np.float64)
        R_rect = stereo_params["R_L"].astype(np.float64)
        K = stereo_params["K_L"].astype(np.float64)
    else:
        P = stereo_params["P_R"].astype(np.float64)
        R_rect = stereo_params["R_R"].astype(np.float64)
        K = stereo_params["K_R"].astype(np.float64)

    P3 = P[:3, :3]
    mapx, mapy = cv2.initUndistortRectifyMap(P3, np.zeros(5), R_rect.T, K, (W, H), cv2.CV_32FC1)
    img_orig = cv2.remap(rectified_img, mapx, mapy, cv2.INTER_NEAREST)
    return img_orig

# ======================================================================================
# Visualization helper for u (overlay, points-only, or both)
# ======================================================================================
def overlay_pointcloud_image(
    img_bgr: np.ndarray,
    u: np.ndarray,
    radius: int = 3,
    alpha: float = 0.6,
    tint_strength: float = 0.7,
    blur: int = 0,
    mode: str = "overlay"       # "overlay" | "points_only" | "both"
):
    """
    Visualize projected points or a pre-rendered point layer on top of an image.

    Modes:
        - "overlay":     return a single image with a red-tinted fog over original
        - "points_only": return only the red-tinted point layer on black
        - "both":        return (overlay_image, points_only_image)

    Args:
        img_bgr (np.ndarray): Base image (H,W,3) BGR.
        u (np.ndarray): Either Nx2 pixel coordinates, or an (H,W,3) mask/layer image.
        radius, alpha, tint_strength, blur, mode: visualization controls.

    Behavior:
        - If u is (H,W,3): treat it as a ready-made "point layer" and alpha blend.
        - If u is (N,2): draw red-tinted points on img_bgr.
    """

    H, W = img_bgr.shape[:2]

    # ✅ Case 1: u 是图像层 (H, W, 3)
    if u.ndim == 3 and u.shape[:2] == (H, W):
        overlay_only = u.copy()
        # 若输入是灰度或单通道，可自动扩展
        if overlay_only.ndim == 2:
            overlay_only = cv2.cvtColor(overlay_only, cv2.COLOR_GRAY2BGR)
        if blur and blur >= 2:
            overlay_only = cv2.GaussianBlur(overlay_only, (blur|1, blur|1), 0)
        if mode == "points_only":
            return overlay_only
        overlay_on_img = cv2.addWeighted(overlay_only, alpha, img_bgr, 1.0, 0)
        if mode == "both":
            return overlay_on_img, overlay_only
        else:
            return overlay_on_img

    # ✅ Case 2: u 是 Nx2 坐标数组
    if u.size == 0:
        return (img_bgr.copy(), np.zeros((H, W, 3), np.uint8)) if mode == "both" \
               else (img_bgr.copy() if mode == "overlay" else np.zeros((H, W, 3), np.uint8))

    uv = np.rint(u).astype(int)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not np.any(inb):
        return (img_bgr.copy(), np.zeros((H, W, 3), np.uint8)) if mode == "both" \
               else (img_bgr.copy() if mode == "overlay" else np.zeros((H, W, 3), np.uint8))
    uv = uv[inb]
    xs, ys = uv[:, 0], uv[:, 1]
    cols = img_bgr[ys, xs].astype(np.float32)

    # Push sampled colors toward red
    s = float(np.clip(tint_strength, 0.0, 1.0))
    red = np.zeros_like(cols); red[:, 2] = 255.0
    tinted = np.clip((1.0 - s) * cols + s * red, 0, 255).astype(np.uint8)

    overlay_only = np.zeros_like(img_bgr, dtype=np.uint8)
    r = max(1, int(radius))
    for (x, y), c in zip(uv, tinted):
        cv2.circle(overlay_only, (int(x), int(y)), r, tuple(int(v) for v in c), -1, cv2.LINE_AA)

    if blur and blur >= 2:
        overlay_only = cv2.GaussianBlur(overlay_only, (blur|1, blur|1), 0)

    if mode == "points_only":
        return overlay_only

    overlay_on_img = cv2.addWeighted(overlay_only, alpha, img_bgr, 1.0, 0)
    if mode == "overlay":
        return overlay_on_img
    elif mode == "both":
        return overlay_on_img, overlay_only
    else:
        raise ValueError("mode must be 'overlay' | 'points_only' | 'both'")

