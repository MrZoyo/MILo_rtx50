from pathlib import Path
import math
import json
import random
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import yaml

from argparse import ArgumentParser

from functional import (
    compute_delaunay_triangulation,
    extract_mesh,
    frustum_cull_mesh,
)
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer.radegs import render_radegs, integrate_radegs
from arguments import PipelineParams
from regularization.regularizer.mesh import (
    initialize_mesh_regularization,
    compute_mesh_regularization,
)
from regularization.sdf.learnable import convert_occupancy_to_sdf
from utils.geometry_utils import flatten_voronoi_features

# DISCOVER-SE 相机轨迹使用 OpenGL 右手坐标系（相机前方为 -Z，向上为 +Y），
# 而 MILo/colmap 渲染管线假设的是前方 +Z、向上 -Y。需要在读入时做一次轴翻转。
OPENGL_TO_COLMAP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


class ManualScene:
    """最小 Scene 封装，提供 mesh regularization 所需的接口。"""

    def __init__(self, cameras: Sequence[Camera]):
        self._train_cameras = list(cameras)

    def getTrainCameras(self, scale: float = 1.0) -> List[Camera]:
        return list(self._train_cameras)

    def getTrainCameras_warn_up(
        self,
        iteration: int,
        warn_until_iter: int,
        scale: float = 1.0,
        scale2: float = 2.0,
    ) -> List[Camera]:
        return list(self._train_cameras)


def build_render_functions(
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
):
    """构建与训练阶段一致的 RaDe-GS 渲染接口，兼顾常规前向和 SDF 重建需求。"""
    def _render(
        view: Camera,
        pc_obj: GaussianModel,
        pipe_obj: PipelineParams,
        bg_color: torch.Tensor,
        *,
        kernel_size: float = 0.0,
        require_coord: bool = False,
        require_depth: bool = True,
    ):
        pkg = render_radegs(
            viewpoint_camera=view,
            pc=pc_obj,
            pipe=pipe_obj,
            bg_color=bg_color,
            kernel_size=kernel_size,
            scaling_modifier=1.0,
            require_coord=require_coord,
            require_depth=require_depth,
        )
        if "area_max" not in pkg:
            pkg["area_max"] = torch.zeros_like(pkg["radii"])
        return pkg

    def render_view(view: Camera):
        return _render(view, gaussians, pipe, background)

    def render_for_sdf(
        view: Camera,
        gaussians_override: Optional[GaussianModel] = None,
        pipeline_override: Optional[PipelineParams] = None,
        background_override: Optional[torch.Tensor] = None,
        kernel_size: float = 0.0,
        require_depth: bool = True,
        require_coord: bool = False,
    ):
        pc_obj = gaussians if gaussians_override is None else gaussians_override
        pipe_obj = pipe if pipeline_override is None else pipeline_override
        bg_color = background if background_override is None else background_override
        pkg = _render(
            view,
            pc_obj,
            pipe_obj,
            bg_color,
            kernel_size=kernel_size,
            require_coord=require_coord,
            require_depth=require_depth,
        )
        return {
            "render": pkg["render"].detach(),
            "median_depth": pkg["median_depth"].detach(),
        }

    return render_view, render_for_sdf


def load_mesh_config(config_name: str) -> Dict[str, Any]:
    """支持直接传文件路径或 configs/mesh/<name>.yaml。"""
    candidate = Path(config_name)
    if not candidate.is_file():
        candidate = Path(__file__).resolve().parent / "configs" / "mesh" / f"{config_name}.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"无法找到 mesh 配置：{config_name}")
    with candidate.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_gaussian_occupancy(gaussians: GaussianModel) -> None:
    """mesh regularization 依赖 9 维 occupancy 网格，此处在推理环境补齐缓冲。"""
    needs_init = (
        not getattr(gaussians, "learn_occupancy", False)
        or not hasattr(gaussians, "_occupancy_shift")
        or gaussians._occupancy_shift.numel() == 0
    )
    if needs_init:
        gaussians.learn_occupancy = True
        base = torch.zeros((gaussians._xyz.shape[0], 9), device=gaussians._xyz.device)
        shift = torch.zeros_like(base)
        gaussians._base_occupancy = nn.Parameter(base.requires_grad_(False), requires_grad=False)
        gaussians._occupancy_shift = nn.Parameter(shift.requires_grad_(True))


def export_mesh_from_state(
    gaussians: GaussianModel,
    mesh_state: Dict[str, Any],
    output_path: Path,
    reference_camera: Optional[Camera] = None,
) -> None:
    """根据当前 mesh_state 导出网格，并可选做视椎裁剪。"""
    gaussian_idx = mesh_state.get("delaunay_xyz_idx")
    delaunay_tets = mesh_state.get("delaunay_tets")

    if delaunay_tets is None:
        delaunay_tets = compute_delaunay_triangulation(
            means=gaussians.get_xyz,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            gaussian_idx=gaussian_idx,
        )

    occupancy = (
        gaussians.get_occupancy
        if gaussian_idx is None
        else gaussians.get_occupancy[gaussian_idx]
    )
    pivots_sdf = convert_occupancy_to_sdf(flatten_voronoi_features(occupancy))

    mesh = extract_mesh(
        delaunay_tets=delaunay_tets,
        pivots_sdf=pivots_sdf,
        means=gaussians.get_xyz,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        gaussian_idx=gaussian_idx,
    )

    mesh_to_save = mesh if reference_camera is None else frustum_cull_mesh(mesh, reference_camera)
    verts = mesh_to_save.verts.detach().cpu().numpy()
    faces = mesh_to_save.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(output_path)


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """将单位四元数转换为 3x3 旋转矩阵。"""
    # 这里显式转换 DISCOVERSE 导出的四元数，确保后续符合 MILo 的旋转约定
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
    # 推理阶段冻结高斯参数，后续循环只做前向评估
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
    # 统一 squeeze 逻辑，防止 Matplotlib 因 shape 异常报错
    depth_np = depth_tensor.detach().cpu().numpy()
    depth_np = np.squeeze(depth_np)
    if depth_np.ndim == 1:
        depth_np = np.expand_dims(depth_np, axis=0)
    return depth_np


def prepare_normals(normal_tensor: torch.Tensor) -> np.ndarray:
    """将法线张量转换为 HxWx3 的 numpy 数组。"""
    # 兼容渲染输出为 (3,H,W) 或 (H,W,3) 的两种格式
    normals_np = normal_tensor.detach().cpu().numpy()
    normals_np = np.squeeze(normals_np)
    if normals_np.ndim == 3 and normals_np.shape[0] == 3:
        normals_np = np.transpose(normals_np, (1, 2, 0))
    if normals_np.ndim == 2:
        normals_np = normals_np[..., None]
    return normals_np


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    """将 [-1,1] 范围的法线向量映射到 [0,1] 以便可视化。"""
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = 0.5 * (normals + 1.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def save_normal_visualization(normal_rgb: np.ndarray, output_path: Path) -> None:
    """保存法线可视化图像。"""
    plt.imsave(output_path, normal_rgb)


def load_cameras_from_json(
    json_path: str,
    image_height: int,
    image_width: int,
    fov_y_deg: float,
) -> List[Camera]:
    """按照 MILo 的视图约定读取相机文件，并进行坐标系转换。"""
    # 自定义读取 DISCOVERSE 风格 JSON，统一转换到 COLMAP 世界->相机坐标
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

        # DISCOVER-SE 的 quaternion/rotation 直接导入后，渲染出来的 PNG 会上下翻转，
        # 说明其前进方向仍是 OpenGL 的 -Z。通过右乘 diag(1,-1,-1) 将其显式转换到
        # MILo/colmap 的坐标系，使得后续投影矩阵与深度图一致。
        rotation_c2w = rotation_c2w @ OPENGL_TO_COLMAP

        if "position" in entry:
            camera_center = np.asarray(entry["position"], dtype=np.float32)
            if camera_center.shape != (3,):
                raise ValueError(f"相机条目 {idx} 的 position 应为 3 维向量，实际为 {camera_center.shape}")
            rotation_w2c = rotation_c2w.T
            translation = (-rotation_w2c @ camera_center).astype(np.float32)
        elif "translation" in entry:
            translation = np.asarray(entry["translation"], dtype=np.float32)
            # 如果 JSON 已直接存储 colmap 风格的 T（即世界到相机），这里假设它与旋转
            # 一样来自 OpenGL 坐标。严格来说也应执行同样的轴变换，但现有数据集只有
            # position 字段；为避免重复转换，这里只做类型检查并保留原值。
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


def save_heatmap(data: np.ndarray, output_path: Path, title: str) -> None:
    """将二维数据保存为热力图，便于直观观察差异。"""
    # 迭代间深度 / 法线差分可视化，快速定位局部变化
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
    parser.add_argument(
        "--depth_loss_weight", type=float, default=1.0, help="深度一致性项权重"
    )
    parser.add_argument(
        "--normal_loss_weight", type=float, default=1.0, help="法线一致性项权重"
    )
    parser.add_argument(
        "--delaunay_reset_interval",
        type=int,
        default=50,
        help="每隔多少次迭代重建一次 Delaunay（<=0 表示每次重建）",
    )
    parser.add_argument(
        "--mesh_config",
        type=str,
        default="verylowres",
        help="mesh 配置名称或路径（默认 verylowres）",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="保存可视化/npz 的间隔，默认与 Delaunay 重建间隔相同",
    )
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

    # 所有输出固定写入 milo/runs/ 下，便于管理实验产物
    base_run_dir = Path(__file__).resolve().parent / "runs"
    output_dir = base_run_dir / args.heatmap_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ply_path = "/home/zoyo/Desktop/MILo_rtx50/milo/data/bridge_clean/yufu_bridge_cleaned.ply"
    camera_poses_json = "/home/zoyo/Desktop/MILo_rtx50/milo/data/bridge_clean/camera_poses_cam1.json"

    gaussians = GaussianModel(sh_degree=0, learn_occupancy=True)
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

    mesh_config = load_mesh_config(args.mesh_config)
    mesh_config["start_iter"] = 0
    mesh_config["stop_iter"] = max(mesh_config.get("stop_iter", args.num_iterations), args.num_iterations)
    mesh_config["mesh_update_interval"] = 1
    mesh_config["delaunay_reset_interval"] = args.delaunay_reset_interval
    # 这里默认沿用 surface 采样以对齐训练阶段；如仅需快速分析，也可以切换为 random 提升速度。
    mesh_config["delaunay_sampling_method"] = "surface"

    scene_wrapper = ManualScene(train_cameras)

    ensure_gaussian_occupancy(gaussians)
    if gaussians.spatial_lr_scale <= 0:
        gaussians.spatial_lr_scale = 1.0
    gaussians.set_occupancy_mode(mesh_config.get("occupancy_mode", "occupancy_shift"))

    render_view, render_for_sdf = build_render_functions(gaussians, pipe, background)

    mesh_renderer, mesh_state = initialize_mesh_regularization(scene_wrapper, mesh_config)
    mesh_state["reset_delaunay_samples"] = True
    mesh_state["reset_sdf_values"] = True

    moving_loss = None
    previous_depth: Dict[int, np.ndarray] = {}
    previous_normals: Dict[int, np.ndarray] = {}
    camera_stack = list(range(len(train_cameras)))
    random.shuffle(camera_stack)
    save_interval = args.save_interval if args.save_interval is not None else args.delaunay_reset_interval
    if save_interval is None or save_interval <= 0:
        save_interval = 1

    for iteration in range(args.num_iterations):
        with torch.no_grad():
            if args.lock_view_index is not None:
                view_index = args.lock_view_index % len(train_cameras)
            else:
                if not camera_stack:
                    camera_stack = list(range(len(train_cameras)))
                    random.shuffle(camera_stack)
                view_index = camera_stack.pop()
            viewpoint = train_cameras[view_index]

            gaussian_render_pkg = render_view(viewpoint)
            mesh_pkg = compute_mesh_regularization(
                iteration=iteration,
                render_pkg=gaussian_render_pkg,
                viewpoint_cam=viewpoint,
                viewpoint_idx=view_index,
                gaussians=gaussians,
                scene=scene_wrapper,
                pipe=pipe,
                background=background,
                kernel_size=0.0,
                config=mesh_config,
                mesh_renderer=mesh_renderer,
                mesh_state=mesh_state,
                render_func=render_for_sdf,
                weight_adjustment=1.0,
                args=None,
                integrate_func=integrate_radegs,
            )
            mesh_state = mesh_pkg["updated_state"]

        mesh_render_pkg = mesh_pkg["mesh_render_pkg"]
        depth_map = prepare_depth_map(mesh_render_pkg["depth"])
        normals_map = prepare_normals(mesh_render_pkg["normals"])
        ply_depth_map = prepare_depth_map(gaussian_render_pkg["median_depth"])
        ply_normals_map = prepare_normals(gaussian_render_pkg["normal"])

        mesh_valid = np.isfinite(depth_map) & (depth_map > 0.0)
        ply_valid = np.isfinite(ply_depth_map) & (ply_depth_map > 0.0)
        overlap_mask = mesh_valid & ply_valid

        depth_loss_value = float(mesh_pkg["mesh_depth_loss"].item())
        normal_loss_value = float(mesh_pkg["mesh_normal_loss"].item())
        occupied_loss = float(mesh_pkg["occupied_centers_loss"].item())
        labels_loss = float(mesh_pkg["occupancy_labels_loss"].item())
        loss_value = float(mesh_pkg["mesh_loss"].item())

        moving_loss = (
            loss_value
            if moving_loss is None
            else args.ma_beta * moving_loss + (1 - args.ma_beta) * loss_value
        )

        def _fmt(value: float) -> str:
            return f"{value:.6f}"

        print(
            "[INFO] Iter {iter:02d} | loss={total} (depth={depth}, normal={normal}) | ma_loss={ma}".format(
                iter=iteration,
                total=_fmt(loss_value),
                depth=_fmt(depth_loss_value),
                normal=_fmt(normal_loss_value),
                ma=f"{moving_loss:.6f}",
            )
        )

        should_save = (save_interval <= 0) or (iteration % save_interval == 0)
        if should_save:
            valid_values: List[np.ndarray] = []
            if mesh_valid.any():
                valid_values.append(depth_map[mesh_valid].reshape(-1))
            if ply_valid.any():
                valid_values.append(ply_depth_map[ply_valid].reshape(-1))
            if valid_values:
                all_valid = np.concatenate(valid_values)
                shared_min = float(all_valid.min())
                shared_max = float(all_valid.max())
            else:
                shared_min, shared_max = 0.0, 1.0

            ply_depth_vis_path = output_dir / f"ply_depth_vis_iter_{iteration:02d}.png"
            plt.imsave(ply_depth_vis_path, ply_depth_map, cmap="viridis", vmin=shared_min, vmax=shared_max)

            depth_vis_path = output_dir / f"depth_vis_iter_{iteration:02d}.png"
            plt.imsave(depth_vis_path, depth_map, cmap="viridis", vmin=shared_min, vmax=shared_max)

            normal_vis_path = output_dir / f"normal_vis_iter_{iteration:02d}.png"
            normals_rgb = normals_to_rgb(normals_map)
            save_normal_visualization(normals_rgb, normal_vis_path)

            ply_normal_vis_path = output_dir / f"ply_normal_vis_iter_{iteration:02d}.png"
            ply_normals_rgb = normals_to_rgb(ply_normals_map)
            save_normal_visualization(ply_normals_rgb, ply_normal_vis_path)

            output_npz = output_dir / f"mesh_render_iter_{iteration:02d}.npz"
            np.savez(
                output_npz,
                depth=depth_map,
                ply_depth=ply_depth_map,
                normals=normals_map,
                normal_vis=normals_rgb,
                ply_normals=ply_normals_map,
                ply_normal_vis=ply_normals_rgb,
                depth_loss=depth_loss_value,
                normal_loss=normal_loss_value,
                occupied_centers_loss=occupied_loss,
                occupancy_labels_loss=labels_loss,
                loss=loss_value,
                moving_loss=moving_loss,
            )

            if overlap_mask.any():
                depth_delta = depth_map - ply_depth_map
                delta_abs = np.abs(depth_delta[overlap_mask])
                diff_mean = float(delta_abs.mean())
                diff_max = float(delta_abs.max())
                diff_rmse = float(np.sqrt(np.mean(depth_delta[overlap_mask] ** 2)))
            else:
                diff_mean = diff_max = diff_rmse = float("nan")

            composite_path = output_dir / f"comparison_iter_{iteration:02d}.png"
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
            ax_ply_depth, ax_mesh_depth = axes[0]
            ax_ply_normals, ax_mesh_normals = axes[1]

            im0 = ax_ply_depth.imshow(ply_depth_map, cmap="viridis", vmin=shared_min, vmax=shared_max)
            ax_ply_depth.axis("off")
            fig.colorbar(im0, ax=ax_ply_depth, fraction=0.046, pad=0.04)

            im1 = ax_mesh_depth.imshow(depth_map, cmap="viridis", vmin=shared_min, vmax=shared_max)
            ax_mesh_depth.axis("off")
            fig.colorbar(im1, ax=ax_mesh_depth, fraction=0.046, pad=0.04)

            ax_ply_normals.imshow(ply_normals_rgb)
            ax_ply_normals.axis("off")

            ax_mesh_normals.imshow(normals_rgb)
            ax_mesh_normals.axis("off")
            info_lines = [
                f"Iteration: {iteration:02d}",
                f"View index: {view_index}",
                f"PLY depth valid px: {int(ply_valid.sum())}",
            ]
            if ply_valid.any():
                info_lines.append(
                    f"  min={float(ply_depth_map[ply_valid].min()):.3f}, "
                    f"max={float(ply_depth_map[ply_valid].max()):.3f}, "
                    f"mean={float(ply_depth_map[ply_valid].mean()):.3f}"
                )
            info_lines.append(f"Mesh depth valid px: {int(mesh_valid.sum())}")
            if mesh_valid.any():
                info_lines.append(
                    f"  min={float(depth_map[mesh_valid].min()):.3f}, "
                    f"max={float(depth_map[mesh_valid].max()):.3f}, "
                    f"mean={float(depth_map[mesh_valid].mean()):.3f}"
                )
            if overlap_mask.any():
                info_lines.append(f"|PLY - Mesh| mean={diff_mean:.3f}, max={diff_max:.3f}, RMSE={diff_rmse:.3f}")
            else:
                info_lines.append("No overlapping valid depth pixels.")
            info_lines.append(
                f"Depth loss={_fmt(depth_loss_value)} (w={args.depth_loss_weight:.2f})"
            )
            info_lines.append(
                f"Normal loss={_fmt(normal_loss_value)} (w={args.normal_loss_weight:.2f})"
            )
            info_lines.append(f"Occupied centers={_fmt(occupied_loss)}")
            info_lines.append(f"Labels loss={_fmt(labels_loss)}")
            info_lines.append(f"Overlap px={int(overlap_mask.sum())}")
            mesh_norm_valid = np.all(np.isfinite(normals_map), axis=-1)
            ply_norm_valid = np.all(np.isfinite(ply_normals_map), axis=-1)
            normal_overlap_mask = overlap_mask & mesh_norm_valid & ply_norm_valid
            info_lines.append(f"Normal px={int(normal_overlap_mask.sum())}")
            fig.suptitle("\n".join(info_lines), fontsize=12, y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(composite_path, dpi=300)
            plt.close(fig)

            if args.lock_view_index is not None:
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

            export_mesh_from_state(
                gaussians=gaussians,
                mesh_state=mesh_state,
                output_path=output_dir / f"mesh_iter_{iteration:02d}.ply",
                reference_camera=viewpoint,
            )

        if args.lock_view_index is not None:
            previous_depth[view_index] = depth_map
            previous_normals[view_index] = normals_map
    print("[INFO] 循环结束，所有结果已写入输出目录。")


if __name__ == "__main__":
    main()
