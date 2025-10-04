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
def transform_pointcloud(pcd, stereo_params, pcd_frame="rectified_left"):
    """
    Transform a point cloud (or Nx3 array) to original left/right camera frames
    and project to pixels uL/uR (with lens distortion).

    Args:
        pcd (open3d.geometry.PointCloud or np.ndarray): 3D points. If generated
            from rectified disparity + Q, it is typically in the rectified-left
            camera frame.
        stereo_params (dict): Must include:
            'K_L','D_L','K_R','D_R','R','T','R_L' (float64 recommended).
        pcd_frame (str): 'rectified_left' (default) or 'original_left'.

    Returns:
        tuple:
            left_pts_cam  (N,3)  points in original-left camera coordinates
            right_pts_cam (N,3)  points in original-right camera coordinates
            uL            (N,2)  pixel coords in original-left image (with distortion)
            uR            (N,2)  pixel coords in original-right image (with distortion)

    Notes:
        - If pcd is in rectified-left coordinates, X_orig_L = R_L^T * X_rect_L.
        - Projection uses cv2.projectPoints with rvec=tvec=0 because points are
          already in each camera’s coordinate system.
        - No Z>0 filtering here (kept same as your function). You can mask later if needed.
    """
    # Extract Nx3 points
    if hasattr(pcd, "points"):
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:                   
        pts = np.asarray(pcd, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 3, "pcd must be (N,3) or an Open3D PointCloud"

    # Calibration (float64)
    K_L = stereo_params['K_L'].astype(np.float64)
    D_L = stereo_params['D_L'].astype(np.float64)
    K_R = stereo_params['K_R'].astype(np.float64)
    D_R = stereo_params['D_R'].astype(np.float64)
    R   = stereo_params['R'  ].astype(np.float64)                # left -> right
    T   = stereo_params['T'  ].astype(np.float64).reshape(3,1)   # (3,1)
    R_L = stereo_params['R_L'].astype(np.float64)                # rectification rotation (left)

    # Rectified-left -> Original-left  (X_orig_L = R_L^T * X_rect_L)
    if pcd_frame == "rectified_left":
        left_orig_all = pts @ R_L.T
    elif pcd_frame == "original_left":
        left_orig_all = pts.copy()
    else:
        raise ValueError("pcd_frame must be 'rectified_left' or 'original_left'")

    # Project to original-left pixels
    uL, _ = cv2.projectPoints(
        left_orig_all.reshape(-1,1,3),
        rvec=np.zeros((3,1), np.float64),
        tvec=np.zeros((3,1), np.float64),
        cameraMatrix=K_L,
        distCoeffs=D_L
    )
    uL = uL.reshape(-1, 2)

    # Original-left -> Original-right (X_R = R X_L + T), then project to pixels
    right_orig_all = (R @ left_orig_all.T).T + T.reshape(1,3)
    uR, _ = cv2.projectPoints(
        right_orig_all.reshape(-1,1,3),
        rvec=np.zeros((3,1), np.float64),
        tvec=np.zeros((3,1), np.float64),
        cameraMatrix=K_R,
        distCoeffs=D_R
    )
    uR = uR.reshape(-1, 2)

    return left_orig_all, right_orig_all, uL, uR

# ======================================================================================
# Visualization helper for u (overlay, points-only, or both)
# ======================================================================================
def overlay_pointcloud_image(
    img_bgr: np.ndarray,
    u: np.ndarray,
    radius: int = 3,            # point radius
    alpha: float = 0.6,         # overlay strength [0..1]
    tint_strength: float = 0.7, # how strongly to push colors toward red [0..1]
    blur: int = 0,              # optional Gaussian blur for the point layer; 0 = off
    mode: str = "overlay"       # "overlay" | "points_only" | "both"
):
    """
    Visualize projected points u on top of an image.

    Modes:
        - "overlay":     return a single image with a red-tinted fog over original
        - "points_only": return only the red-tinted point layer on black
        - "both":        return (overlay_image, points_only_image)

    Args/Returns:
        img_bgr (np.ndarray): Base image (H,W,3) BGR.
        u (np.ndarray): Nx2 pixel coordinates.
        radius/alpha/tint_strength/blur/mode: visualization controls.
        Returns either a single image or a tuple of two images depending on mode.

    Notes:
        - Colors for each point are sampled from the base image at u, then pushed
          toward red by 'tint_strength', then optionally blurred and alpha-composited.
        - Coordinates outside image bounds are ignored.
        - No disk I/O is performed; use cv2.imshow / cv2.imwrite as needed outside.
    """

    H, W = img_bgr.shape[:2]
    if u.size == 0:
        return (img_bgr.copy(), np.zeros((H,W,3),np.uint8)) if mode=="both" \
               else (img_bgr.copy() if mode=="overlay" else np.zeros((H,W,3),np.uint8))

    # Bound check & color sampling
    uv = np.rint(u).astype(int)
    inb = (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    if not np.any(inb):
        return (img_bgr.copy(), np.zeros((H,W,3),np.uint8)) if mode=="both" \
               else (img_bgr.copy() if mode=="overlay" else np.zeros((H,W,3),np.uint8))
    uv = uv[inb]
    xs, ys = uv[:,0], uv[:,1]
    cols = img_bgr[ys, xs].astype(np.float32)  # BGR from image

    # Push sampled colors toward red
    s = float(np.clip(tint_strength, 0.0, 1.0))
    red = np.zeros_like(cols); red[:,2] = 255.0
    tinted = np.clip((1.0 - s) * cols + s * red, 0, 255).astype(np.uint8)

    # Draw a pure point layer (black background)
    overlay_only = np.zeros_like(img_bgr, dtype=np.uint8)
    r = max(1, int(radius))
    for (x, y), c in zip(uv, tinted):
        cv2.circle(overlay_only, (int(x), int(y)), r, tuple(int(v) for v in c), -1, cv2.LINE_AA)

    # Optional blur for a softer "fog" look
    if blur and blur >= 2:
        overlay_only = cv2.GaussianBlur(overlay_only, (blur|1, blur|1), 0)

    if mode == "points_only":
        return overlay_only

    # Alpha blend with base image
    a = float(np.clip(alpha, 0.0, 1.0))
    overlay_on_img = cv2.addWeighted(overlay_only, a, img_bgr, 1.0, 0)

    if mode == "overlay":
        return overlay_on_img
    elif mode == "both":
        return overlay_on_img, overlay_only
    else:
        raise ValueError("mode must be 'overlay' | 'points_only' | 'both'")

