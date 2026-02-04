#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sh_dc_to_rgb(dc):
    rgb = dc * C0 + 0.5
    return np.clip(rgb, 0.0, 1.0)


def quat_to_rot(q):
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    q = q / norm
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def sample_unit_ball(rng, count):
    vec = rng.normal(size=(count, 3)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    radius = rng.random(size=(count, 1)).astype(np.float32) ** (1.0 / 3.0)
    return vec * radius


def build_rotation_matrices(rots):
    rotations = np.empty((rots.shape[0], 3, 3), dtype=np.float32)
    for i in range(rots.shape[0]):
        rotations[i] = quat_to_rot(rots[i])
    return rotations


def build_spatial_hash(xyz, max_radii, bbox_min, cell_size):
    inv = 1.0 / cell_size
    grid = {}
    for i in range(xyz.shape[0]):
        r = max_radii[i]
        lo = np.floor((xyz[i] - r - bbox_min) * inv).astype(np.int64)
        hi = np.floor((xyz[i] + r - bbox_min) * inv).astype(np.int64)
        for ix in range(lo[0], hi[0] + 1):
            for iy in range(lo[1], hi[1] + 1):
                for iz in range(lo[2], hi[2] + 1):
                    key = (int(ix), int(iy), int(iz))
                    grid.setdefault(key, []).append(i)
    return grid


def point_in_any_gaussian(points, xyz, radii, rotations, max_radii, bbox_min, cell_size, grid):
    hits = np.zeros(points.shape[0], dtype=bool)
    inv = 1.0 / cell_size
    for i in range(points.shape[0]):
        if hits[i]:
            continue
        cell = np.floor((points[i] - bbox_min) * inv).astype(np.int64)
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        candidates = grid.get(key)
        if not candidates:
            continue
        for j in candidates:
            delta = points[i] - xyz[j]
            if np.dot(delta, delta) > max_radii[j] ** 2:
                continue
            local = delta @ rotations[j]
            scaled = local / radii[j]
            if np.dot(scaled, scaled) <= 1.0:
                hits[i] = True
                break
    return hits


def load_gaussians(path):
    plydata = PlyData.read(path)
    verts = plydata["vertex"]

    xyz = np.stack((verts["x"], verts["y"], verts["z"]), axis=1).astype(np.float32)
    opacity_raw = np.asarray(verts["opacity"]).astype(np.float32)
    opacity = sigmoid(opacity_raw)

    f_dc = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    if "f_dc_0" in verts.data.dtype.names:
        f_dc[:, 0] = np.asarray(verts["f_dc_0"], dtype=np.float32)
        f_dc[:, 1] = np.asarray(verts["f_dc_1"], dtype=np.float32)
        f_dc[:, 2] = np.asarray(verts["f_dc_2"], dtype=np.float32)

    scale_names = [n for n in verts.data.dtype.names if n.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.stack([np.asarray(verts[n], dtype=np.float32) for n in scale_names], axis=1)
    sigma = np.exp(scales)

    rot_names = [n for n in verts.data.dtype.names if n.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.stack([np.asarray(verts[n], dtype=np.float32) for n in rot_names], axis=1)

    return xyz, sigma, rots, opacity, f_dc


def build_pointcloud(
    xyz,
    sigma,
    rots,
    opacity,
    f_dc,
    sigma_multiplier,
    points_per_unit_volume,
    total_points,
    min_points,
    max_points,
    max_total_points,
    min_opacity,
    use_opacity,
    seed,
    with_color,
):
    rng = np.random.default_rng(seed)

    if min_opacity > 0.0:
        keep = opacity >= min_opacity
        xyz, sigma, rots, opacity, f_dc = (
            xyz[keep],
            sigma[keep],
            rots[keep],
            opacity[keep],
            f_dc[keep],
        )

    radii = sigma_multiplier * sigma
    volumes = (4.0 / 3.0) * math.pi * radii[:, 0] * radii[:, 1] * radii[:, 2]
    weights = opacity if use_opacity else np.ones_like(opacity)

    weighted_volume = volumes * weights
    if total_points is not None:
        denom = np.sum(weighted_volume)
        points_per_unit_volume = total_points / denom if denom > 0 else 0.0

    expected_counts = points_per_unit_volume * weighted_volume

    if max_total_points is not None and expected_counts.sum() > max_total_points:
        expected_counts *= max_total_points / expected_counts.sum()

    base = np.floor(expected_counts).astype(np.int64)
    frac = expected_counts - base
    base += (rng.random(size=frac.shape) < frac).astype(np.int64)

    if min_points is not None:
        base = np.maximum(base, min_points)
    if max_points is not None:
        base = np.minimum(base, max_points)

    all_points = []
    all_colors = []
    for i in range(xyz.shape[0]):
        count = int(base[i])
        if count <= 0:
            continue
        local = sample_unit_ball(rng, count)
        local *= radii[i]
        R = quat_to_rot(rots[i])
        world = local @ R.T
        world += xyz[i]
        all_points.append(world)
        if with_color:
            rgb = sh_dc_to_rgb(f_dc[i])
            rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
            all_colors.append(np.repeat(rgb_u8[None, :], count, axis=0))

    if not all_points:
        return np.empty((0, 3), dtype=np.float32), None

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0) if with_color else None
    return points, colors


def build_pointcloud_density(
    xyz,
    sigma,
    rots,
    f_dc,
    sigma_multiplier,
    total_points,
    min_opacity,
    use_opacity,
    opacity,
    seed,
    with_color,
):
    rng = np.random.default_rng(seed)

    if min_opacity > 0.0:
        keep = opacity >= min_opacity
        xyz, sigma, rots, opacity, f_dc = (
            xyz[keep],
            sigma[keep],
            rots[keep],
            opacity[keep],
            f_dc[keep],
        )

    if xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), None

    rotations = build_rotation_matrices(rots)
    radii = sigma_multiplier * sigma
    max_radii = np.max(radii, axis=1)

    bbox_min = np.min(xyz - max_radii[:, None], axis=0)
    bbox_max = np.max(xyz + max_radii[:, None], axis=0)
    cell_size = np.median(max_radii[max_radii > 0]) if np.any(max_radii > 0) else 1.0
    cell_size = max(cell_size, 1e-6)
    grid = build_spatial_hash(xyz, max_radii, bbox_min, cell_size)

    total_points = 1_000_000 if total_points is None else total_points

    points_out = []
    colors_out = []
    batch_size = max(4096, min(65536, total_points // 4))
    while sum(p.shape[0] for p in points_out) < total_points:
        remaining = total_points - sum(p.shape[0] for p in points_out)
        batch = min(batch_size, remaining * 4)
        samples = rng.uniform(bbox_min, bbox_max, size=(batch, 3)).astype(np.float32)

        hits = point_in_any_gaussian(samples, xyz, radii, rotations, max_radii, bbox_min, cell_size, grid)
        kept = samples[hits]
        if kept.shape[0] == 0:
            continue

        if kept.shape[0] > remaining:
            kept = kept[:remaining]

        points_out.append(kept)
        if with_color:
            # Color is taken from the nearest gaussian center.
            d2 = np.sum((kept[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(d2, axis=1)
            rgb = sh_dc_to_rgb(f_dc[nearest])
            colors_out.append((rgb * 255.0).round().astype(np.uint8))

    points = np.concatenate(points_out, axis=0)
    colors = np.concatenate(colors_out, axis=0) if with_color else None
    return points, colors


def save_pointcloud(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if colors is None:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        data = np.empty(points.shape[0], dtype=dtype)
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    else:
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        data = np.empty(points.shape[0], dtype=dtype)
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
        data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert Gaussian splat .ply to a more uniform point cloud."
    )
    parser.add_argument("input_ply", type=Path, help="Input gaussian .ply file")
    parser.add_argument("output_ply", type=Path, help="Output point cloud .ply file")
    parser.add_argument(
        "--mode",
        choices=["volume", "density"],
        default="volume",
        help="Sampling mode: fill each gaussian (volume) or uniform density via bbox rejection (density)",
    )
    parser.add_argument(
        "--sigma-multiplier",
        type=float,
        default=2.0,
        help="Extent of each gaussian in sigma units",
    )
    parser.add_argument(
        "--points-per-unit-volume",
        type=float,
        default=1500.0,
        help="Target density in points per unit volume",
    )
    parser.add_argument(
        "--total-points",
        type=int,
        default=None,
        help="Override density to reach this total count",
    )
    parser.add_argument("--min-points", type=int, default=0, help="Min points per gaussian")
    parser.add_argument("--max-points", type=int, default=None, help="Max points per gaussian")
    parser.add_argument(
        "--max-total-points",
        type=int,
        default=2_000_000,
        help="Cap total points after sampling",
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.0,
        help="Drop gaussians with opacity below this",
    )
    parser.add_argument(
        "--no-opacity-weight",
        action="store_true",
        help="Do not weight sampling by opacity",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Do not write RGB color attributes",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--natedebug', action='store_true', help='activate vscode debugger')

    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.natedebug:
        import debugpy
        debugpy.listen(5678)
        debugpy.wait_for_client() 
        
    xyz, sigma, rots, opacity, f_dc = load_gaussians(args.input_ply)
    if args.mode == "density":
        points, colors = build_pointcloud_density(
            xyz=xyz,
            sigma=sigma,
            rots=rots,
            f_dc=f_dc,
            sigma_multiplier=args.sigma_multiplier,
            total_points=args.total_points,
            min_opacity=args.min_opacity,
            use_opacity=not args.no_opacity_weight,
            opacity=opacity,
            seed=args.seed,
            with_color=not args.no_color,
        )
    else:
        points, colors = build_pointcloud(
            xyz=xyz,
            sigma=sigma,
            rots=rots,
            opacity=opacity,
            f_dc=f_dc,
            sigma_multiplier=args.sigma_multiplier,
            points_per_unit_volume=args.points_per_unit_volume,
            total_points=args.total_points,
            min_points=args.min_points,
            max_points=args.max_points,
            max_total_points=args.max_total_points,
            min_opacity=args.min_opacity,
            use_opacity=not args.no_opacity_weight,
            seed=args.seed,
            with_color=not args.no_color,
        )

    save_pointcloud(args.output_ply, points, colors)


if __name__ == "__main__":
    main()
