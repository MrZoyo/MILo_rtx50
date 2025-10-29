#!/usr/bin/env python3
"""Iteratively refine learnable occupancy (SDF) while keeping Gaussian geometry fixed."""

import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from arguments import PipelineParams
from gaussian_renderer.radegs import integrate_radegs
from ply2mesh import (
    ManualScene,
    load_cameras_from_json,
    freeze_gaussian_rigid_parameters,
    build_render_functions,
    load_mesh_config_file,
    export_mesh_from_gaussians,
)
from regularization.regularizer.mesh import initialize_mesh_regularization, compute_mesh_regularization
from regularization.sdf.learnable import convert_occupancy_to_sdf
from scene.gaussian_model import GaussianModel
from utils.geometry_utils import flatten_voronoi_features


def ensure_learnable_occupancy(gaussians: GaussianModel) -> None:
    """Ensure occupancy buffers exist and only the shift is trainable."""
    if not gaussians.learn_occupancy or not hasattr(gaussians, "_occupancy_shift"):
        device = gaussians._xyz.device
        n_pts = gaussians._xyz.shape[0]
        base = torch.zeros((n_pts, 9), device=device)
        shift = torch.zeros_like(base)
        gaussians.learn_occupancy = True
        gaussians._base_occupancy = torch.nn.Parameter(base.requires_grad_(False), requires_grad=False)
        gaussians._occupancy_shift = torch.nn.Parameter(shift.requires_grad_(True))
    gaussians.set_occupancy_mode("occupancy_shift")
    gaussians._occupancy_shift.requires_grad_(True)


def export_iteration_state(
    iteration: int,
    gaussians: GaussianModel,
    mesh_state: dict,
    output_dir: str,
    reference_camera=None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mesh_path = os.path.join(output_dir, f"mesh_iter_{iteration:05d}.ply")
    ply_path = os.path.join(output_dir, f"gaussians_iter_{iteration:05d}.ply")

    export_mesh_from_gaussians(
        gaussians=gaussians,
        mesh_state=mesh_state,
        output_path=mesh_path,
        reference_camera=reference_camera,
    )
    gaussians.save_ply(ply_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Occupancy-only refinement from a pretrained Gaussian PLY.")
    parser.add_argument("--ply_path", type=str, required=True, help="Input Gaussian PLY (assumed geometrically correct).")
    parser.add_argument("--camera_poses", type=str, required=True, help="JSON with camera poses matching the scene.")
    parser.add_argument("--mesh_config", type=str, default="default", help="Mesh regularization config name.")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of optimization steps.")
    parser.add_argument("--occupancy_lr", type=float, default=0.001, help="Learning rate for occupancy shift.")
    parser.add_argument(
        "--mesh_loss_weight",
        type=float,
        default=5.0,
        help="Global weight applied to the mesh regularization loss.",
    )
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval.")
    parser.add_argument("--export_interval", type=int, default=1000, help="Mesh export interval (0 disables periodic export).")
    parser.add_argument("--output_dir", type=str, default="./occupancy_refine_output", help="Directory to store outputs.")
    parser.add_argument("--fov_y", type=float, default=75.0, help="Vertical field-of-view in degrees.")
    parser.add_argument("--image_width", type=int, default=1280, help="Rendered image width.")
    parser.add_argument("--image_height", type=int, default=720, help="Rendered image height.")
    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Background color used for rendering (RGB).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--mesh_start_iter", type=int, default=1, help="Iteration at which mesh regularization starts.")
    parser.add_argument(
        "--mesh_stop_iter",
        type=int,
        default=None,
        help="Iteration after which mesh regularization stops (defaults to total iterations).",
    )
    parser.add_argument("--warn_until_iter", type=int, default=3000, help="Warmup iterations for surface sampling.")
    parser.add_argument(
        "--imp_metric",
        type=str,
        default="outdoor",
        choices=["outdoor", "indoor"],
        help="Importance metric used for surface sampling.",
    )
    parser.add_argument(
        "--cull_on_export",
        action="store_true",
        help="Frustum cull meshes using the first camera before export.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for occupancy refinement.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cameras = load_cameras_from_json(
        json_path=args.camera_poses,
        image_height=args.image_height,
        image_width=args.image_width,
        fov_y_deg=args.fov_y,
    )
    print(f"[INFO] Loaded {len(cameras)} cameras from {args.camera_poses}.")

    scene = ManualScene(cameras)

    mesh_config = load_mesh_config_file(args.mesh_config)
    mesh_config["start_iter"] = max(1, args.mesh_start_iter)
    if args.mesh_stop_iter is not None:
        mesh_config["stop_iter"] = args.mesh_stop_iter
    else:
        mesh_config["stop_iter"] = max(mesh_config.get("stop_iter", args.iterations), args.iterations)

    pipe_parser = argparse.ArgumentParser()
    pipe = PipelineParams(pipe_parser)
    background = torch.tensor(args.background, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        sh_degree=0,
        use_mip_filter=False,
        learn_occupancy=True,
        use_appearance_network=False,
    )
    gaussians.load_ply(args.ply_path)
    print(f"[INFO] Loaded {gaussians._xyz.shape[0]} Gaussians from {args.ply_path}.")

    ensure_learnable_occupancy(gaussians)

    if gaussians.spatial_lr_scale <= 0:
        gaussians.spatial_lr_scale = 1.0

    gaussians.init_culling(len(cameras))
    gaussians.set_occupancy_mode(mesh_config.get("occupancy_mode", "occupancy_shift"))
    freeze_gaussian_rigid_parameters(gaussians)

    optimizer = torch.optim.Adam([gaussians._occupancy_shift], lr=args.occupancy_lr)

    render_view, render_for_sdf = build_render_functions(gaussians, pipe, background)
    mesh_renderer, mesh_state = initialize_mesh_regularization(scene, mesh_config)
    mesh_state["reset_delaunay_samples"] = True
    mesh_state["reset_sdf_values"] = True

    runtime_args = SimpleNamespace(
        warn_until_iter=args.warn_until_iter,
        imp_metric=args.imp_metric,
        depth_reinit_iter=getattr(args, "depth_reinit_iter", args.warn_until_iter),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    ema_loss = None
    pending_view_indices: list[int] = []

    # Iterate through all cameras without replacement; reshuffle when one pass finishes.
    for iteration in range(1, args.iterations + 1):
        if not pending_view_indices:
            pending_view_indices = list(range(len(cameras)))
            random.shuffle(pending_view_indices)

        view_idx = pending_view_indices.pop()
        viewpoint = cameras[view_idx]

        render_pkg = render_view(viewpoint)

        mesh_pkg = compute_mesh_regularization(
            iteration=iteration,
            render_pkg=render_pkg,
            viewpoint_cam=viewpoint,
            viewpoint_idx=view_idx,
            gaussians=gaussians,
            scene=scene,
            pipe=pipe,
            background=background,
            kernel_size=0.0,
            config=mesh_config,
            mesh_renderer=mesh_renderer,
            mesh_state=mesh_state,
            render_func=render_for_sdf,
            weight_adjustment=100.0 / max(args.iterations, 1),
            args=runtime_args,
            integrate_func=integrate_radegs,
        )
        mesh_state = mesh_pkg["updated_state"]

        if mesh_pkg.get("mesh_triangles") is not None and mesh_pkg["mesh_triangles"].numel() == 0:
            print(f"[WARNING] Empty mesh at iteration {iteration}; skipping optimizer step.")
            continue

        raw_mesh_loss = mesh_pkg["mesh_loss"]
        loss = args.mesh_loss_weight * raw_mesh_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        raw_loss_value = float(raw_mesh_loss.item())
        ema_loss = loss_value if ema_loss is None else (0.9 * ema_loss + 0.1 * loss_value)

        if iteration % args.log_interval == 0 or iteration == 1:
            mesh_depth = mesh_pkg["mesh_depth_loss"].item()
            mesh_normal = mesh_pkg["mesh_normal_loss"].item()
            occupied_centers = mesh_pkg["occupied_centers_loss"].item()
            occupancy_labels = mesh_pkg["occupancy_labels_loss"].item()

            current_occ = torch.sigmoid(gaussians._base_occupancy + gaussians._occupancy_shift)
            pivots_sdf = convert_occupancy_to_sdf(flatten_voronoi_features(current_occ))
            sdf_mean = float(pivots_sdf.mean().item())

            print(
                "[Iter {iter:05d}] loss={loss:.6f} ema={ema:.6f} depth={depth:.6f} "
                "normal={normal:.6f} occ_centers={centers:.6f} labels={labels:.6f} "
                "sdf_mean={sdf_mean:.6f} mesh_raw={raw_mesh:.6f}".format(
                    iter=iteration,
                    loss=loss_value,
                    ema=ema_loss,
                    depth=mesh_depth,
                    normal=mesh_normal,
                    centers=occupied_centers,
                    labels=occupancy_labels,
                    sdf_mean=sdf_mean,
                    raw_mesh=raw_loss_value,
                )
            )

        if args.export_interval > 0 and iteration % args.export_interval == 0:
            export_iteration_state(
                iteration=iteration,
                gaussians=gaussians,
                mesh_state=mesh_state,
                output_dir=args.output_dir,
                reference_camera=cameras[0] if args.cull_on_export else None,
            )

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    export_iteration_state(
        iteration=args.iterations,
        gaussians=gaussians,
        mesh_state=mesh_state,
        output_dir=final_dir,
        reference_camera=cameras[0] if args.cull_on_export else None,
    )

    print(f"[INFO] Occupancy refinement completed. Results saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
