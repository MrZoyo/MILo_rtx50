from pathlib import Path
import math
import json
import random
from typing import List, Sequence

import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from functional import (
    sample_gaussians_on_surface,
    compute_initial_sdf_values,
    compute_delaunay_triangulation,
    extract_mesh,
    frustum_cull_mesh,
)
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.mesh import MeshRasterizer, MeshRenderer
from gaussian_renderer.radegs import render_radegs
from arguments import PipelineParams


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """将单位四元数转换为 3x3 旋转矩阵。"""
    # 新增：这里显式转换 DISCOVERSE 导出的四元数，确保后续符合 MILo 的旋转约定
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"四元数需要包含 4 个分量，当前形状为 {q.shape}")
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rotation = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    return rotation


def freeze_gaussian_model(model: GaussianModel) -> None:
    """显式关闭高斯模型中参数的梯度。"""
    # 新增：推理阶段冻结高斯参数，后续循环只做前向评估
    tensor_attrs = [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    ]
    for attr in tensor_attrs:
        value = getattr(model, attr, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)


def prepare_depth_map(depth_tensor: torch.Tensor) -> np.ndarray:
    """将深度张量转为二维 numpy 数组。"""
    # 新增：统一 squeeze 逻辑，防止 Matplotlib 因 shape 异常报错
    depth_np = depth_tensor.detach().cpu().numpy()
    depth_np = np.squeeze(depth_np)
    if depth_np.ndim == 1:
        depth_np = np.expand_dims(depth_np, axis=0)
    return depth_np


def prepare_normals(normal_tensor: torch.Tensor) -> np.ndarray:
    """将法线张量转换为 HxWx3 的 numpy 数组。"""
    # 新增：兼容渲染输出为 (3,H,W) 或 (H,W,3) 的两种格式
    normals_np = normal_tensor.detach().cpu().numpy()
    normals_np = np.squeeze(normals_np)
    if normals_np.ndim == 3 and normals_np.shape[0] == 3:
        normals_np = np.transpose(normals_np, (1, 2, 0))
    if normals_np.ndim == 2:
        normals_np = normals_np[..., None]
    return normals_np


def load_cameras_from_json(
    json_path: str,
    image_height: int,
    image_width: int,
    fov_y_deg: float,
) -> List[Camera]:
    """按照 MILo 的视图约定读取相机文件，并进行坐标系转换。"""
    # 新增：自定义读取 DISCOVERSE 风格 JSON，统一转换到 COLMAP 世界->相机坐标
    pose_path = Path(json_path)
    if not pose_path.is_file():
        raise FileNotFoundError(f"未找到相机 JSON：{json_path}")

    with pose_path.open("r", encoding="utf-8") as fh:
        camera_list = json.load(fh)

    if isinstance(camera_list, dict):
        for key in ("frames", "poses", "camera_poses"):
            if key in camera_list and isinstance(camera_list[key], list):
                camera_list = camera_list[key]
                break
        else:
            raise ValueError(f"{json_path} 中的 JSON 结构不包含可识别的相机列表。")

    if not isinstance(camera_list, list) or not camera_list:
        raise ValueError(f"{json_path} 中没有有效的相机条目。")

    fov_y = math.radians(fov_y_deg)
    aspect_ratio = image_width / image_height
    fov_x = 2.0 * math.atan(aspect_ratio * math.tan(fov_y * 0.5))

    cameras: List[Camera] = []
    for idx, entry in enumerate(camera_list):
        if "quaternion" in entry:
            rotation_c2w = quaternion_to_rotation_matrix(entry["quaternion"]).astype(np.float32)
        elif "rotation" in entry:
            rotation_c2w = np.asarray(entry["rotation"], dtype=np.float32)
        else:
            raise KeyError(f"相机条目 {idx} 未提供 quaternion 或 rotation。")

        if rotation_c2w.shape != (3, 3):
            raise ValueError(f"相机条目 {idx} 的旋转矩阵形状应为 (3,3)，实际为 {rotation_c2w.shape}")

        if "position" in entry:
            camera_center = np.asarray(entry["position"], dtype=np.float32)
            if camera_center.shape != (3,):
                raise ValueError(f"相机条目 {idx} 的 position 应为 3 维向量，实际为 {camera_center.shape}")
            rotation_w2c = rotation_c2w.T
            translation = (-rotation_w2c @ camera_center).astype(np.float32)
        elif "translation" in entry:
            translation = np.asarray(entry["translation"], dtype=np.float32)
        elif "tvec" in entry:
            translation = np.asarray(entry["tvec"], dtype=np.float32)
        else:
            raise KeyError(f"相机条目 {idx} 未提供 position/translation/tvec 信息。")

        if translation.shape != (3,):
            raise ValueError(f"相机条目 {idx} 的平移向量应为长度 3，实际为 {translation.shape}")

        image_name = (
            entry.get("name")
            or entry.get("image_name")
            or entry.get("img_name")
            or f"view_{idx:04d}"
        )

        camera = Camera(
            colmap_id=str(idx),
            R=rotation_c2w,
            T=translation,
            FoVx=fov_x,
            FoVy=fov_y,
            image=torch.zeros(3, image_height, image_width),
            gt_alpha_mask=None,
            image_name=image_name,
            uid=idx,
            data_device="cuda",
        )
        cameras.append(camera)
    return cameras


def compute_gaussian_mesh_loss(
    mesh_vertices: torch.Tensor,
    gaussian_means: torch.Tensor,
    max_samples: int,
) -> torch.Tensor:
    """
    估算高斯中心与网格顶点之间的平均欧氏距离。
    新增：通过随机抽样避免 O(N^2) 计算，提供稳定的 mesh 与高斯差异指标。
    """
    if mesh_vertices.numel() == 0 or gaussian_means.numel() == 0:
        device = mesh_vertices.device if mesh_vertices.numel() else gaussian_means.device
        return torch.zeros(1, device=device, dtype=torch.float32)

    device = mesh_vertices.device
    gaussian_means = gaussian_means.to(device)

    mesh_count = mesh_vertices.shape[0]
    gaussian_count = gaussian_means.shape[0]
    sample_size = min(max_samples, mesh_count, gaussian_count)

    if sample_size == 0:
        return torch.zeros(1, device=device, dtype=torch.float32)

    if mesh_count > sample_size:
        mesh_indices = torch.randperm(mesh_count, device=device)[:sample_size]
        mesh_samples = mesh_vertices[mesh_indices]
    else:
        mesh_samples = mesh_vertices

    if gaussian_count > sample_size:
        gaussian_indices = torch.randperm(gaussian_count, device=device)[:sample_size]
        gaussian_samples = gaussian_means[gaussian_indices]
    else:
        gaussian_samples = gaussian_means

    pairing_count = min(mesh_samples.shape[0], gaussian_samples.shape[0])
    if pairing_count == 0:
        return torch.zeros(1, device=device, dtype=torch.float32)

    mesh_samples = mesh_samples[:pairing_count]
    gaussian_samples = gaussian_samples[:pairing_count]
    distances = torch.norm(mesh_samples - gaussian_samples, dim=1)
    return distances.mean()


def save_heatmap(data: np.ndarray, output_path: Path, title: str) -> None:
    """将二维数据保存为热力图，便于直观观察差异。"""
    # 新增：迭代间深度 / 法线差分可视化，快速定位局部变化
    plt.figure(figsize=(6, 4))
    plt.imshow(data, cmap="inferno")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = ArgumentParser(description="桥梁场景高斯到网格迭代分析脚本")
    parser.add_argument("--num_iterations", type=int, default=5, help="执行循环的次数")
    parser.add_argument("--ma_beta", type=float, default=0.8, help="loss 滑动平均系数")
    parser.add_argument("--max_loss_samples", type=int, default=5000, help="loss 采样点上限")
    parser.add_argument(
        "--heatmap_dir",
        type=str,
        default="yufu2mesh_outputs",
        help="保存热力图等输出的目录",
    )
    parser.add_argument("--seed", type=int, default=0, help="控制随机性的种子")
    parser.add_argument(
        "--lock_view_index",
        type=int,
        default=None,
        help="固定视角索引，仅在指定时输出热力图",
    )
    pipe = PipelineParams(parser)
    args = parser.parse_args()

    pipe.debug = getattr(args, "debug", False)

    # 新增：所有输出固定写入 milo/runs/ 下，便于管理实验产物
    base_run_dir = Path(__file__).resolve().parent / "runs"
    output_dir = base_run_dir / args.heatmap_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ply_path = "/home/zoyo/Desktop/MILo_rtx50/milo/data/Bridge/yufu_bridge_cleaned.ply"
    camera_poses_json = "/home/zoyo/Desktop/MILo_rtx50/milo/data/Bridge/camera_poses_cam1.json"

    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(ply_path)
    freeze_gaussian_model(gaussians)

    height = 720
    width = 1280
    fov_y_deg = 75.0

    train_cameras = load_cameras_from_json(
        json_path=camera_poses_json,
        image_height=height,
        image_width=width,
        fov_y_deg=fov_y_deg,
    )
    print(f"[INFO] 成功加载 {len(train_cameras)} 个相机视角。")

    background = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    def render_func(view):
        # 新增：封装 RaDe-GS 渲染接口，用于 SDF 初始化
        render_pkg = render_radegs(
            viewpoint_camera=view,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            kernel_size=0.0,
            scaling_modifier=1.0,
            require_coord=False,
            require_depth=True,
        )
        return {"render": render_pkg["render"], "depth": render_pkg["median_depth"]}

    renderer = MeshRenderer(MeshRasterizer(cameras=train_cameras))

    moving_loss = None
    previous_depth = {}
    previous_normals = {}
    camera_stack = list(range(len(train_cameras)))
    random.shuffle(camera_stack)

    # 新增：循环执行多次网格提取与渲染，统计 loss 及深度/法线变化
    for iteration in range(args.num_iterations):
        with torch.no_grad():
            means = gaussians.get_xyz
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation
            opacities = gaussians.get_opacity

            surface_gaussians_idx = sample_gaussians_on_surface(
                views=train_cameras,
                means=means,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                n_max_samples=600_000,
                scene_type="outdoor",
            )

            initial_pivots_sdf = compute_initial_sdf_values(
                views=train_cameras,
                render_func=render_func,
                means=means,
                scales=scales,
                rotations=rotations,
                gaussian_idx=surface_gaussians_idx,
            )

            delaunay_tets = compute_delaunay_triangulation(
                means=means,
                scales=scales,
                rotations=rotations,
                gaussian_idx=surface_gaussians_idx,
            )

            mesh = extract_mesh(
                delaunay_tets=delaunay_tets,
                pivots_sdf=initial_pivots_sdf,
                means=means,
                scales=scales,
                rotations=rotations,
                gaussian_idx=surface_gaussians_idx,
            )

            if args.lock_view_index is not None:
                # 新增：允许通过参数锁定视角，确保热图比较有意义
                view_index = args.lock_view_index % len(train_cameras)
            else:
                if not camera_stack:
                    camera_stack = list(range(len(train_cameras)))
                    random.shuffle(camera_stack)
                view_index = camera_stack.pop()
            refined_mesh = frustum_cull_mesh(mesh, train_cameras[view_index])

            mesh_render_pkg = renderer(
                refined_mesh,
                cam_idx=view_index,
                return_depth=True,
                return_normals=True,
            )

        loss_tensor = compute_gaussian_mesh_loss(refined_mesh.verts, means[surface_gaussians_idx], max_samples=args.max_loss_samples)
        loss_value = float(loss_tensor.item())
        moving_loss = (
            loss_value if moving_loss is None else args.ma_beta * moving_loss + (1 - args.ma_beta) * loss_value
        )
        print(f"[INFO] Iter {iteration:02d} | loss={loss_value:.6f} | ma_loss={moving_loss:.6f}")

        depth_map = prepare_depth_map(mesh_render_pkg["depth"])
        normals_map = prepare_normals(mesh_render_pkg["normals"])

        output_npz = output_dir / f"mesh_render_iter_{iteration:02d}.npz"
        np.savez(output_npz, depth=depth_map, normals=normals_map, loss=loss_value, moving_loss=moving_loss)

        if args.lock_view_index is not None:
            # 新增：仅在锁定视角时对比上一帧，避免跨视角差异
            if view_index in previous_depth:
                depth_diff = np.abs(depth_map - previous_depth[view_index])
                save_heatmap(depth_diff, output_dir / f"depth_diff_iter_{iteration:02d}.png", f"Depth Δ iter {iteration}")
            if view_index in previous_normals:
                normal_delta = normals_map - previous_normals[view_index]
                if normal_delta.ndim == 3:
                    normal_diff = np.linalg.norm(normal_delta, axis=-1)
                else:
                    normal_diff = np.abs(normal_delta)
                save_heatmap(
                    normal_diff,
                    output_dir / f"normal_diff_iter_{iteration:02d}.png",
                    f"Normal Δ iter {iteration}",
                )
            previous_depth[view_index] = depth_map
            previous_normals[view_index] = normals_map

        refined_vertices = refined_mesh.verts.detach().cpu().numpy()
        refined_faces = refined_mesh.faces.detach().cpu().numpy()
        refined_mesh_obj = trimesh.Trimesh(vertices=refined_vertices, faces=refined_faces)
        refined_mesh_obj.export(output_dir / f"refined_mesh_iter_{iteration:02d}.ply")

        vertices = mesh.verts.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_obj.export(output_dir / f"mesh_iter_{iteration:02d}.ply")

    print("[INFO] 循环结束，所有结果已写入输出目录。")


if __name__ == "__main__":
    main()
