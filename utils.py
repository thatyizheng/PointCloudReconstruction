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
    tint_strength: float = 0.7,   # 0=不偏红, 1=强烈偏红
    blur: int = 0,
    mode: str = "overlay",        # "overlay" | "points_only" | "both"
    color_src: np.ndarray = None, # 颜色采样来源；默认用 img_bgr
    preview_bg: str = "black"     # points_only 预览背景: "black" | "gray"
):
    """
    半透明点层，颜色来自 color_src 对应像素，并向红色偏移 (tint_strength)。
    - 若 u 为 (N,2)：从 color_src[x,y] 取色，在黑底画点，然后做 alpha 合成。
    - 若 u 为 (H,W,3)：视为“点的掩膜图”(非零处为点)；同样从 color_src 取色并着色、合成。
    - overlay：把点层 alpha 合成到 img_bgr；points_only：返回与预览底色合成后的半透明层；
      both：同时返回 overlay 和 points_only 预览。
    """
    H, W = img_bgr.shape[:2]
    if color_src is None:
        color_src = img_bgr
    assert color_src.shape[:2] == (H, W), "color_src must match img_bgr size"

    a = float(np.clip(alpha, 0.0, 1.0))
    a255 = int(a * 255)

    def make_preview_bg():
        if preview_bg == "gray":
            return np.full((H, W, 3), 64, np.uint8)
        return np.zeros((H, W, 3), np.uint8)

    # -------- 构建“着色后的前景层（BGR）” 及 alpha mask ----------
    layer_bgr = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)

    if u.ndim == 2 and u.shape[1] == 2:
        # Nx2 坐标：逐点从 color_src 取色 → 向红偏移 → 画点
        uv = np.rint(u).astype(int)
        inb = (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        if np.any(inb):
            uv = uv[inb]
            xs, ys = uv[:,0], uv[:,1]
            cols = color_src[ys, xs].astype(np.float32)
            # toward red
            s = float(np.clip(tint_strength, 0.0, 1.0))
            red = np.zeros_like(cols); red[:,2] = 255.0
            tinted = np.clip((1.0 - s) * cols + s * red, 0, 255).astype(np.uint8)
            r = max(1, int(radius))
            for (x, y), c in zip(uv, tinted):
                cv2.circle(layer_bgr, (int(x), int(y)), r, tuple(int(v) for v in c), -1, cv2.LINE_AA)
                cv2.circle(mask,      (int(x), int(y)), r, 255, -1, cv2.LINE_AA)
    else:
        # (H,W,3) 图：把非零像素当作“点的掩膜”，在这些处用 color_src 取色并向红偏移
        assert u.ndim == 3 and u.shape[:2] == (H, W), "u must be Nx2 or (H,W,3)"
        mask_bool = np.any(u != 0, axis=2)
        idx = np.where(mask_bool)
        if idx[0].size > 0:
            cols = color_src[idx].astype(np.float32)
            s = float(np.clip(tint_strength, 0.0, 1.0))
            red = np.zeros_like(cols); red[:,2] = 255.0
            tinted = np.clip((1.0 - s) * cols + s * red, 0, 255).astype(np.uint8)
            layer_bgr[idx] = tinted
            mask[idx] = 255

    # 可选柔化边缘
    if blur and blur >= 2:
        k = blur | 1
        layer_bgr = cv2.GaussianBlur(layer_bgr, (k,k), 0)
        mask      = cv2.GaussianBlur(mask, (k,k), 0)

    # 组 BGRA（真实 alpha）
    layer_bgra = np.dstack([layer_bgr, (mask * a / 255.0).astype(np.float32) * 255]).astype(np.uint8)

    # 手写 alpha 合成（OpenCV 不直接用 alpha，需要自己算）
    def alpha_compose(fg_bgra, bg_bgr):
        fg_rgb = fg_bgra[...,:3].astype(np.float32)
        fg_a   = (fg_bgra[...,3:4].astype(np.float32)) / 255.0
        bg_rgb = bg_bgr.astype(np.float32)
        out = fg_rgb * fg_a + bg_rgb * (1.0 - fg_a)
        return np.clip(out, 0, 255).astype(np.uint8)

    overlay_img = alpha_compose(layer_bgra, img_bgr)
    preview_bg_bgr = make_preview_bg()
    points_preview = alpha_compose(layer_bgra, preview_bg_bgr)

    if mode == "overlay":
        return overlay_img
    elif mode == "points_only":
        return points_preview  # 已与预览底色合成，肉眼可见半透明（颜色来自 color_src 并偏红）
    elif mode == "both":
        return overlay_img, points_preview
    else:
        raise ValueError("mode must be 'overlay' | 'points_only' | 'both'")



################################################################################################################3
from scipy.spatial.transform import Rotation as R
def quaternion_to_transformation(quaternion):
    """
    Convert a quaternion + translation to a 4x4 homogeneous transformation matrix.

    Note:
        This function expects a mapping-like input (e.g., dict, pandas Series, or 1-row DataFrame slice)
        containing keys: 'q0','qx','qy','qz','tx','ty','tz'. The quaternion is treated with scalar first.

    Args:
        quaternion: A mapping with fields q0, qx, qy, qz, tx, ty, tz (floats).

    Returns:
        np.ndarray: 4x4 homogeneous transform with rotation from the quaternion and translation (tx,ty,tz).
    """
    from scipy.spatial.transform import Rotation as R

    # Extract quaternion and translation; ensure float
    q0, qx, qy, qz = float(quaternion['q0']), float(quaternion['qx']), float(quaternion['qy']), float(quaternion['qz'])
    tx, ty, tz = float(quaternion['tx']), float(quaternion['ty']), float(quaternion['tz'])

    # Quaternion to rotation (scalar-first convention)
    rotation = R.from_quat([q0, qx, qy, qz], scalar_first=True)
    rotation_matrix = rotation.as_matrix()  # 3x3 rotation

    # Build 4x4 homogeneous transform
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]
    return transformation_matrix

#################################################################################################

def transform_CTMR(stl_path, stl_files, OT_to_BF_qt, BF_to_EN, pcd, CT_to_OT):
    OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
    OT_to_CT = np.linalg.inv(CT_to_OT)

    CM_to_EN = np.array([
        [1,  0,  0, 0],
        [0,  -1,  0, 0],
        [0,  0,  -1, 0],
        [0,  0,  0, 1]
    ])

    O3D_to_CM = np.array([
        [1,  0,  0, 0],
        [0,  1,  0, 0],  # Y 轴反了
        [0,  0,  1, 0],
        [0,  0,  0, 1]
    ])

    BF_to_OT = np.linalg.inv(OT_to_BF)
    BF_to_CT = np.dot(BF_to_OT, OT_to_CT)
    EN_to_BF = np.linalg.inv(BF_to_EN)
    CM_to_BF = np.dot(CM_to_EN, EN_to_BF)
    CM_to_CT = np.dot(CM_to_BF, BF_to_CT)

    O3D_to_CT = np.dot(O3D_to_CM, CM_to_CT)

    meshes_CT = []
    for file in stl_files:
        path = stl_path + '/' + file
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        mesh.transform(O3D_to_CT)   # 将 STLs（O3D 坐标假定）搬到 CT 坐标
        meshes_CT.append(mesh)

    return pcd, meshes_CT

##########################################################################################################
def visualize_pcd_mesh(
    pcd: o3d.geometry.PointCloud,
    meshes_CT: list,                      # [o3d.geometry.TriangleMesh, ...]
    *,
    window_name: str = "PCD + CT Viewer",
    width: int = 1280,
    height: int = 800,
    # 初始视角
    front=(0.0, 0.0, 0.001),
    lookat=(0.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0),
    zoom: float = 0.7
):
    """
    交互说明（按键）:
      1: 仅显示 PCD
      2: 仅显示 Mesh(CT)
      3: 同时显示 PCD + Mesh
      T: 在 PCD 与 Mesh 之间快速切换
      R: 复位到初始视角
      H: 打印帮助
      Q/ESC: 退出
    鼠标：左键旋转、Shift+左键平移、滚轮缩放
    """

    assert isinstance(meshes_CT, (list, tuple)) and all(isinstance(m, o3d.geometry.TriangleMesh) for m in meshes_CT)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name, width=width, height=height, visible=True)

    state = {
        "show_pcd": True,
        "show_mesh": True,
        "init_cam": None,   # 初始相机参数（用于复位）
    }

    def _print_help():
        print(
            "\n[Hotkeys]\n"
            "  1 : show PCD only\n"
            "  2 : show Mesh only\n"
            "  3 : show PCD + Mesh\n"
            "  T : toggle between PCD / Mesh\n"
            "  R : reset to initial view\n"
            "  H : help\n"
            "  Q / ESC : quit\n"
            "Mouse: L-drag rotate, Shift+L-drag pan, wheel zoom\n"
        )

    def _apply_view(vis_):
        vc = vis_.get_view_control()
        vc.set_front(front)
        vc.set_lookat(lookat)
        vc.set_up(up)
        vc.set_zoom(zoom)

    def _snapshot_init_cam(vis_):
        vc = vis_.get_view_control()
        state["init_cam"] = vc.convert_to_pinhole_camera_parameters()

    def _reset_view(vis_):
        if state["init_cam"] is not None:
            vc = vis_.get_view_control()
            vc.convert_from_pinhole_camera_parameters(state["init_cam"])
            vis_.update_renderer()

    def _refresh(vis_):
        vis_.clear_geometries()
        if state["show_pcd"] and pcd is not None:
            vis_.add_geometry(pcd)
        if state["show_mesh"]:
            for m in meshes_CT:
                vis_.add_geometry(m)
        vis_.update_renderer()

    # ---- 首次添加并设置初始视角 ----
    _refresh(vis)
    _apply_view(vis)
    _snapshot_init_cam(vis)
    _print_help()

    # ---- 注册热键 ----
    def cb_show_pcd(vis_):
        state["show_pcd"], state["show_mesh"] = True, False
        _refresh(vis_)
        return False

    def cb_show_mesh(vis_):
        state["show_pcd"], state["show_mesh"] = False, True
        _refresh(vis_)
        return False

    def cb_show_both(vis_):
        state["show_pcd"], state["show_mesh"] = True, True
        _refresh(vis_)
        return False

    def cb_toggle(vis_):
        # 快速切换：PCD<->Mesh；如果都开，则只关 Mesh
        if state["show_pcd"] and state["show_mesh"]:
            state["show_mesh"] = False
        else:
            state["show_pcd"], state["show_mesh"] = state["show_mesh"], state["show_pcd"]
        _refresh(vis_)
        return False

    def cb_reset_view(vis_):
        _reset_view(vis_)
        return False

    def cb_help(vis_):
        _print_help()
        return False

    def cb_quit(vis_):
        vis_.close()
        return False
    
    def cb_reset_colors(vis_):
        ro = vis_.get_render_option()
        # 点云颜色用 geometry 自带颜色（或白色），不要按高度上色
        ro.point_color_option = o3d.visualization.PointColorOption.Default
        # 网格用顶点/面颜色（不要用法线/纹理着色）
        ro.mesh_color_option  = o3d.visualization.MeshColorOption.Color
        vis_.update_renderer()
        return True  # 表示“我处理了这个按键”，阻止默认行为

    vis.register_key_callback(ord('0'), cb_reset_colors)
    vis.register_key_callback(ord('4'), cb_reset_colors)
    vis.register_key_callback(ord('1'), cb_show_pcd)
    vis.register_key_callback(ord('2'), cb_show_mesh)
    vis.register_key_callback(ord('3'), cb_show_both)
    vis.register_key_callback(ord('T'), cb_toggle)
    vis.register_key_callback(ord('t'), cb_toggle)
    vis.register_key_callback(ord('R'), cb_reset_view)
    vis.register_key_callback(ord('r'), cb_reset_view)
    vis.register_key_callback(ord('H'), cb_help)
    vis.register_key_callback(ord('h'), cb_help)
    vis.register_key_callback(ord('Q'), cb_quit)
    vis.register_key_callback(256,    cb_quit)  # ESC

    # 进入事件循环
    vis.run()
    vis.destroy_window()
