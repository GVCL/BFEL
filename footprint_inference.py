"""
Footprint Inference Module
==========================
Extracts building footprints from airborne LiDAR point clouds using the
Seed–Refine–Snap methodology.

Feature layout (4 channels):
    [0] chord_dev      — chord deviation curvature indicator
    [1] linearity      — PCA-based local linearity
    [2] density        — neighbour count within radius
    [3] signal_prior   — Gaussian-smoothed corner peaks
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import MultiLineString, Polygon, Point
from shapely.ops import polygonize
import open3d as o3d
import geopandas as gpd

# Set base directory for publication standard
BASE_DIR = "/mnt/d/Saqib/Footprint_Extraction/DIF/Code"



# ─────────────────────────────────────────────────────────────────────────────
# DATASET GEOMETRY CONFIGS  (must match training notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
REG_CFG = dict(
    dp_tolerance=0.45,
    corner_thresh_deg=25.0,
    min_wall_pts=5,
    area_tol=0.15,
    snap_tol_deg=10.0,
    curve_var_thresh=0.01,
)

AHN_GEOMETRY_CFG = dict(
    dataset_name="AHN",
    expected_density_range=(10.0, 20.0),
    expected_spacing_m=0.37,
    alpha_factor=0.82,
    alpha_k=8,
    alpha=0.82 / 0.37,
    chord_stride=3,
    lin_radius=0.37,
    den_radius=0.37,
    sp_sigma=2.5,
    sp_height=0.20,
    sp_distance=8,
    point_count=150,
    reg_cfg=dict(REG_CFG),
)

VAIHINGEN_GEOMETRY_CFG = dict(
    dataset_name="VAIHINGEN",
    expected_density_range=(4.0, 8.0),
    expected_spacing_m=0.50,
    alpha_factor=0.55,
    alpha_k=9,
    alpha=0.55 / 0.50,
    chord_stride=3,
    lin_radius=0.3,
    den_radius=0.3,
    sp_sigma=2.5,
    sp_height=0.15,
    sp_distance=8,
    point_count=150,
    reg_cfg=dict(REG_CFG),
)

DATASET_CFGS = {
    "AHN": AHN_GEOMETRY_CFG,
    "VAIHINGEN": VAIHINGEN_GEOMETRY_CFG,
}

FEATURE_NAMES = ["chord_dev", "linearity", "density", "signal_prior"]
NUM_FEATURES = len(FEATURE_NAMES)  # 4


# ─────────────────────────────────────────────────────────────────────────────
# 1. GEOMETRY & FEATURE EXTRACTION  
# ─────────────────────────────────────────────────────────────────────────────

def compute_tangent_angles(pts):
    """Compute per-point tangent angle from cyclic neighbours."""
    d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    return np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)


def estimate_nn_spacing(pts, k=8):
    """Mean k-NN spacing of a 2D point set."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return float("nan")
    k = int(max(1, min(k, len(pts) - 1)))
    d = cKDTree(pts).query(pts, k=k + 1)[0][:, 1:]
    return float(np.mean(d))


def estimate_alpha(pts, k=8, alpha_factor=0.82, spacing=None, return_spacing=False):
    """
    Alpha is estimated from local point spacing.

    Matches training notebook formula exactly:
        alpha = alpha_factor / spacing

    Smaller spacing → larger alpha → tighter hull.
    """
    if spacing is None:
        spacing = estimate_nn_spacing(pts, k=k)
    if not np.isfinite(spacing) or spacing <= 1e-12:
        spacing = 1.0
    alpha = float(alpha_factor / spacing)
    return (alpha, float(spacing)) if return_spacing else alpha


def alpha_shape(pts, alpha):
    """Compute the alpha shape of a 2D point set."""
    if len(pts) < 3:
        return None
    tri = Delaunay(pts)
    s = tri.simplices
    pa, pb, pc = pts[s[:, 0]], pts[s[:, 1]], pts[s[:, 2]]
    a = np.linalg.norm(pb - pa, axis=1)
    b = np.linalg.norm(pc - pb, axis=1)
    c = np.linalg.norm(pa - pc, axis=1)
    sp = (a + b + c) / 2
    area = np.maximum(sp * (sp - a) * (sp - b) * (sp - c), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = np.where(area > 0, a * b * c / (4 * np.sqrt(np.maximum(area, 1e-20))), np.inf)
    keep = s[cr < 1.0 / alpha]
    if len(keep) == 0:
        return None
    all_edges = np.sort(np.vstack([keep[:, [0, 1]], keep[:, [1, 2]], keep[:, [2, 0]]]), axis=1)
    ev = np.ascontiguousarray(all_edges).view(np.dtype((np.void, all_edges.dtype.itemsize * 2))).ravel()
    uniq, cnt = np.unique(ev, return_counts=True)
    boundary = uniq[cnt == 1].view(np.int32).reshape(-1, 2)
    polys = list(polygonize(MultiLineString([(pts[e[0]], pts[e[1]]) for e in boundary])))
    return max(polys, key=lambda p: p.area) if polys else None


def sample_hull(hull, n):
    """Uniformly sample n points along the hull exterior ring."""
    coords = np.array(hull.exterior.coords)[:-1]
    K = len(coords)
    segs = np.linalg.norm(np.diff(np.vstack([coords, coords[0]]), axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    t_vals = np.linspace(0, cum[-1], n, endpoint=False)
    seg_i = np.searchsorted(cum[1:], t_vals, side="right").clip(0, K - 1)
    t_loc = np.clip((t_vals - cum[seg_i]) / (segs[seg_i] + 1e-12), 0.0, 1.0)
    return (coords[seg_i] + t_loc[:, None] * (coords[(seg_i + 1) % K] - coords[seg_i])).astype(np.float64)


def chord_deviation(pts, stride=3):
    """Chord-based curvature indicator."""
    n = len(pts)
    A = pts[(np.arange(n) - stride) % n]
    B = pts[(np.arange(n) + stride) % n]
    AB = B - A
    AP = pts - A
    cross = np.abs(AB[:, 0] * AP[:, 1] - AB[:, 1] * AP[:, 0])
    denom = np.linalg.norm(AB, axis=1)
    return np.divide(cross, denom, where=denom > 1e-8, out=np.zeros(n, dtype=np.float32)).astype(np.float32)


def linearity_2d(hull_pts, raw_pts, radius=1.0):
    """PCA-based local linearity at each hull point."""
    tree = cKDTree(raw_pts)
    out = np.zeros(len(hull_pts), dtype=np.float32)
    neighbours = tree.query_ball_point(hull_pts, r=radius)
    for i, idx in enumerate(neighbours):
        if len(idx) < 3:
            continue
        c = raw_pts[idx] - raw_pts[idx].mean(axis=0)
        ev = np.linalg.eigvalsh(c.T @ c)
        if ev[-1] > 1e-12:
            out[i] = float((ev[-1] - ev[-2]) / ev[-1])
    return out


def density_support(hull_pts, raw_pts, radius=1.0):
    """Count of raw points within radius of each hull point."""
    tree = cKDTree(raw_pts)
    counts = np.array([len(tree.query_ball_point(p, r=radius)) for p in hull_pts], dtype=np.float32)
    return counts


def signal_prior(pts, sigma=2.5, height=0.20, distance=8):
    """Gaussian-smoothed turning-angle peak detector — binary corner signal."""
    d1 = np.roll(pts, -1, axis=0) - pts
    d2 = pts - np.roll(pts, 1, axis=0)
    dot = (d1 * d2).sum(axis=1)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    theta = np.abs(np.arctan2(np.abs(cross), dot))
    theta_s = gaussian_filter1d(theta, sigma=sigma, mode="wrap")
    peaks, _ = find_peaks(theta_s, height=height, distance=distance)
    out = np.zeros(len(pts), dtype=np.float32)
    out[peaks] = 1.0
    return out


def build_feature_matrix_srsp(
    hull_pts,
    raw_pts,
    geometry_cfg=None,
    chord_stride=None,
    lin_radius=None,
    den_radius=None,
    sp_sigma=None,
    sp_height=None,
    sp_distance=None,
):
    """
    Build the 4-channel SRSP feature tensor.

    IMPORTANT: This matches the training notebook exactly — 4 features only:
        [chord_dev, linearity, density, signal_prior]

    No tangent encoding channels are included (those were incorrectly
    added in a prior version of this module).
    """
    geometry_cfg = geometry_cfg or {}
    chord_stride = int(geometry_cfg.get("chord_stride", chord_stride if chord_stride is not None else 3))
    lin_radius   = float(geometry_cfg.get("lin_radius",   lin_radius   if lin_radius   is not None else 1.0))
    den_radius   = float(geometry_cfg.get("den_radius",   den_radius   if den_radius   is not None else 1.0))
    sp_sigma     = float(geometry_cfg.get("sp_sigma",     sp_sigma     if sp_sigma     is not None else 2.5))
    sp_height    = float(geometry_cfg.get("sp_height",    sp_height    if sp_height    is not None else 0.20))
    sp_distance  = int(geometry_cfg.get("sp_distance",   sp_distance   if sp_distance   is not None else 8))

    chd = chord_deviation(hull_pts, stride=chord_stride)
    lin = linearity_2d(hull_pts, raw_pts, radius=lin_radius)
    den = density_support(hull_pts, raw_pts, radius=den_radius)
    sp  = signal_prior(hull_pts, sigma=sp_sigma, height=sp_height, distance=sp_distance)
    return np.stack([chd, lin, den, sp], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE  
# ─────────────────────────────────────────────────────────────────────────────

class CircPad(nn.Module):
    """Circular padding for 1D convolutions — preserves polygon topology."""
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    def forward(self, x):
        return F.pad(x, (self.pad, self.pad), mode="circular")


class DSResBlock(nn.Module):
    """Depthwise-separable residual Conv1d with circular padding."""
    def __init__(self, ch, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            CircPad(pad),
            nn.Conv1d(ch, ch, kernel, dilation=dilation, groups=ch, bias=False),
            nn.Conv1d(ch, ch, 1, bias=False),
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


class BoundaryRefineCNN(nn.Module):
    """
    GeoRefineNet — lightweight 1D CNN for boundary refinement.

    Input:  (B, in_ch, N)   — in_ch=4 for the full feature set
    Output: (B, 2,     N)   — local tangent-normal displacement (Δt, Δn)
    """
    def __init__(self, in_ch=4, base_ch=64, n_blocks=4, kernel=5, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            CircPad((kernel - 1) // 2),
            nn.Conv1d(in_ch, base_ch, kernel, bias=False),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            DSResBlock(base_ch, kernel=kernel, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch // 2, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(base_ch // 2, 2, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GEOMETRIC REGULARIZER  (PCA-anchored version from training notebook)
# ─────────────────────────────────────────────────────────────────────────────

def _close(pts):
    """Ensure the polygon ring is closed."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    if np.allclose(pts[0], pts[-1]):
        return pts
    return np.vstack([pts, pts[0]])


def geometric_regularizer(
    pts,
    dp_tolerance=0.45,
    corner_thresh_deg=25.0,
    min_wall_pts=5,
    area_tol=0.15,
    snap_tol_deg=10.0,
    curve_var_thresh=0.01,
):
    """
    PCA-anchored orthogonal regularizer — ported from training notebook.

    Estimates the footprint direction from wall geometry, snaps edge chunks
    to the two orthogonal axes, merges only very close parallel lines, and
    rebuilds the polygon from exact 90-degree intersections.
    """
    try:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 3:
            return pts
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            return pts

        poly_in = Polygon(pts)
        if not poly_in.is_valid:
            poly_in = poly_in.buffer(0)
        if poly_in.is_empty:
            return pts

        coords = np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]
        if len(coords) < 3:
            return pts

        # --- Helper functions ---
        def _dedupe_consecutive(arr, eps=1e-9):
            arr = np.asarray(arr, dtype=np.float64)
            if len(arr) == 0:
                return arr
            keep = [0]
            for i in range(1, len(arr)):
                if np.linalg.norm(arr[i] - arr[keep[-1]]) > eps:
                    keep.append(i)
            out = arr[keep]
            if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= eps:
                out = out[:-1]
            return out

        def _pca_basis(arr):
            ctr = arr.mean(axis=0)
            centered = arr - ctr
            cov = np.cov(centered.T) if len(arr) > 1 else np.eye(2)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            axis1 = eigvecs[:, order[0]].astype(np.float64)
            axis2 = eigvecs[:, order[1]].astype(np.float64)
            axis1 /= max(np.linalg.norm(axis1), 1e-12)
            axis2 /= max(np.linalg.norm(axis2), 1e-12)
            if axis1[0] * axis2[1] - axis1[1] * axis2[0] < 0:
                axis2 = -axis2
            return ctr, axis1, axis2

        def _dominant_edge_basis(arr):
            edges = np.roll(arr, -1, axis=0) - arr
            lengths = np.linalg.norm(edges, axis=1)
            mask = lengths > 1e-9
            if not np.any(mask):
                return _pca_basis(arr)[1:]
            edges = edges[mask]
            lengths = lengths[mask]
            dirs = edges / lengths[:, None]
            ang = np.mod(np.arctan2(dirs[:, 1], dirs[:, 0]), np.pi)
            c = float(np.sum(lengths * np.cos(2.0 * ang)))
            s = float(np.sum(lengths * np.sin(2.0 * ang)))
            theta = 0.5 * np.arctan2(s, c)
            axis1 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
            axis1 /= max(np.linalg.norm(axis1), 1e-12)
            axis2 = np.array([-axis1[1], axis1[0]], dtype=np.float64)
            return axis1, axis2

        def _to_local(arr, ctr, axis1, axis2):
            rel = arr - ctr
            u = rel @ axis1
            v = rel @ axis2
            return np.stack([u, v], axis=1)

        def _from_local(local_pts, ctr, axis1, axis2):
            local_pts = np.asarray(local_pts, dtype=np.float64)
            return ctr + local_pts[:, [0]] * axis1[None, :] + local_pts[:, [1]] * axis2[None, :]

        def _cluster_1d(values, weights, tol):
            values = np.asarray(values, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)
            if len(values) == 0:
                return []
            order = np.argsort(values)
            values = values[order]
            weights = weights[order]
            clusters = []
            cur_v = [float(values[0])]
            cur_w = [float(weights[0])]
            for v, w in zip(values[1:], weights[1:]):
                if abs(float(v) - cur_v[-1]) <= tol:
                    cur_v.append(float(v))
                    cur_w.append(float(w))
                else:
                    clusters.append((np.array(cur_v, dtype=np.float64), np.array(cur_w, dtype=np.float64)))
                    cur_v = [float(v)]
                    cur_w = [float(w)]
            clusters.append((np.array(cur_v, dtype=np.float64), np.array(cur_w, dtype=np.float64)))
            return clusters

        def _axis_angle_deg(edge_dir, axis):
            edge_dir = edge_dir / max(np.linalg.norm(edge_dir), 1e-12)
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            dot = float(np.clip(abs(np.dot(edge_dir, axis)), 0.0, 1.0))
            return float(np.degrees(np.arccos(dot)))

        # --- Main regularization logic ---
        coords = _dedupe_consecutive(coords)
        if len(coords) < 3:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        ctr_pca, pca_axis1, pca_axis2 = _pca_basis(coords)
        edge_axis1, edge_axis2 = _dominant_edge_basis(coords)

        # Use PCA when it agrees with wall geometry; otherwise anchor to
        # the dominant wall directions to avoid rotating the footprint core.
        if abs(float(np.dot(pca_axis1, edge_axis1))) >= np.cos(np.deg2rad(6.0)):
            axis1, axis2 = pca_axis1, pca_axis2
        else:
            axis1, axis2 = edge_axis1, edge_axis2

        ctr = coords.mean(axis=0)
        local = _to_local(coords, ctr, axis1, axis2)

        n = len(coords)
        if n < 3:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        bbox_diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        merge_tol = max(bbox_diag * 0.002, min(float(dp_tolerance) * 0.25, bbox_diag * 0.02))
        chunk_size = 10 if n >= 10 else n

        # Build edge records
        edge_records = []
        for i in range(n):
            a = coords[i]
            b = coords[(i + 1) % n]
            d = b - a
            seg_len = float(np.linalg.norm(d))
            if seg_len <= 1e-10:
                continue
            dir_vec = d / seg_len
            ang1 = _axis_angle_deg(dir_vec, axis1)
            ang2 = _axis_angle_deg(dir_vec, axis2)
            axis_idx = 0 if ang1 <= ang2 else 1
            mid_local = 0.5 * (local[i] + local[(i + 1) % n])
            const = float(mid_local[1] if axis_idx == 0 else mid_local[0])
            edge_records.append({"axis": axis_idx, "const": const, "length": seg_len})

        if len(edge_records) < 3:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        # Chunk the boundary for conservative line fits
        chunk_candidates = {0: [], 1: []}
        for start in range(0, n, chunk_size):
            idxs = [((start + j) % n) for j in range(min(chunk_size, n))]
            # Guard: skip indices that are out of range for edge_records
            valid_idxs = [k for k in idxs if k < len(edge_records)]
            for axis_idx in (0, 1):
                vals = [edge_records[k]["const"] for k in valid_idxs if edge_records[k]["axis"] == axis_idx]
                if len(vals) == 0:
                    continue
                wts = [edge_records[k]["length"] for k in valid_idxs if edge_records[k]["axis"] == axis_idx]
                for group_vals, group_wts in _cluster_1d(vals, wts, merge_tol):
                    if len(group_vals) == 0:
                        continue
                    chunk_candidates[axis_idx].append(
                        (float(np.average(group_vals, weights=group_wts)), float(np.sum(group_wts)))
                    )

        global_centers = {}
        for axis_idx in (0, 1):
            vals = np.array([v for v, _ in chunk_candidates[axis_idx]], dtype=np.float64)
            weights = np.array([w for _, w in chunk_candidates[axis_idx]], dtype=np.float64)
            if len(vals) == 0:
                vals = np.array([rec["const"] for rec in edge_records if rec["axis"] == axis_idx], dtype=np.float64)
                weights = np.ones(len(vals), dtype=np.float64)
            if len(vals) == 0:
                return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]
            clusters = _cluster_1d(vals, weights, merge_tol)
            centers = []
            for group_vals, group_wts in clusters:
                if np.sum(group_wts) <= 0:
                    continue
                centers.append(float(np.average(group_vals, weights=group_wts)))
            if len(centers) == 0:
                centers = [float(np.average(vals, weights=weights))]
            global_centers[axis_idx] = centers

        def _nearest_center(axis_idx, value):
            centers = np.asarray(global_centers[axis_idx], dtype=np.float64)
            if len(centers) == 0:
                return float(value)
            return float(centers[np.argmin(np.abs(centers - value))])

        snapped_seq = [(rec["axis"], _nearest_center(rec["axis"], rec["const"])) for rec in edge_records]

        # Collapse immediately repeated directions
        runs = []
        for axis_idx, const in snapped_seq:
            if not runs or runs[-1][0] != axis_idx:
                runs.append([axis_idx, [const]])
            else:
                runs[-1][1].append(const)

        if len(runs) > 1 and runs[0][0] == runs[-1][0]:
            runs[0][1] = runs[-1][1] + runs[0][1]
            runs.pop()

        runs = [(axis_idx, float(np.mean(vals))) for axis_idx, vals in runs if len(vals) > 0]
        if len(runs) < 4:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        if len(runs) >= 2:
            cleaned = [runs[0]]
            for axis_idx, const in runs[1:]:
                if cleaned[-1][0] == axis_idx:
                    cleaned[-1] = (axis_idx, float((cleaned[-1][1] + const) / 2.0))
                else:
                    cleaned.append((axis_idx, const))
            if len(cleaned) > 1 and cleaned[0][0] == cleaned[-1][0]:
                cleaned[0] = (cleaned[0][0], float((cleaned[0][1] + cleaned[-1][1]) / 2.0))
                cleaned.pop()
            runs = cleaned

        if len(runs) < 4:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        # Rebuild vertices from adjacent orthogonal line intersections
        local_vertices = []
        for i in range(len(runs)):
            prev_axis, prev_c = runs[i - 1]
            curr_axis, curr_c = runs[i]
            if prev_axis == curr_axis:
                return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]
            u = prev_c if prev_axis == 1 else curr_c if curr_axis == 1 else None
            v = prev_c if prev_axis == 0 else curr_c if curr_axis == 0 else None
            if u is None or v is None:
                return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]
            local_vertices.append([u, v])

        local_vertices = np.asarray(local_vertices, dtype=np.float64)
        out = _from_local(local_vertices, ctr, axis1, axis2)
        out = _dedupe_consecutive(out)

        if len(out) < 3:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        poly_out = Polygon(out)
        if not poly_out.is_valid:
            poly_fix = poly_out.buffer(0)
            if poly_fix.is_empty:
                return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]
            poly_out = poly_fix

        if poly_out.is_empty or poly_out.area < 1e-6:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        area_change = abs(poly_out.area - poly_in.area) / max(poly_in.area, 1e-6)
        if area_change > area_tol:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        out = np.asarray(poly_out.exterior.coords, dtype=np.float64)[:-1]
        out = _dedupe_consecutive(out)
        if len(out) < 3:
            return np.asarray(poly_in.exterior.coords, dtype=np.float64)[:-1]

        return out
    except Exception:
        return np.asarray(pts, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALIZATION STATS LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_norm_stats(path):
    """
    Load normalization statistics from a pickle file.

    Handles both formats:
      - Wrapped:   {"stats": {"feat_mean": ..., ...}, "meta": {...}}
      - Unwrapped: {"feat_mean": ..., "feat_std": ..., ...}

    Returns the stats dict with keys: feat_mean, feat_std, targ_mean, targ_std.
    """
    import sys
    import numpy
    
    # Ensure backward compatibility for numpy serialization.
    if 'numpy._core' not in sys.modules and hasattr(numpy, 'core'):
        sys.modules['numpy._core'] = numpy.core
    if 'numpy._core.numeric' not in sys.modules and hasattr(numpy, 'core'):
        if hasattr(numpy.core, 'numeric'):
            sys.modules['numpy._core.numeric'] = numpy.core.numeric
            
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "stats" in payload:
        return payload["stats"]
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def local_to_global(shifts_local, phi):
    """Convert local tangent-normal displacements to global XY."""
    c = np.cos(phi)
    s = np.sin(phi)
    dx = shifts_local[:, 0] * c - shifts_local[:, 1] * s
    dy = shifts_local[:, 0] * s + shifts_local[:, 1] * c
    return np.stack([dx, dy], axis=1).astype(np.float32)


@torch.no_grad()
def extract_single_footprint(
    raw_pts,
    model,
    stats,
    device,
    num_points=150,
    use_postproc=True,
    geometry_cfg=None,
    reg_cfg_override=None,
):
    """
    Given a raw cluster of 3D/2D points representing a single building,
    extracts the final refined 2D Polygon footprint.

    Parameters
    ----------
    raw_pts : np.ndarray
        Raw point cloud (Nx2 or Nx3). Only XY are used.
    model : BoundaryRefineCNN or None
        Trained refinement model. If None, returns alpha-shape seed only.
    stats : dict or None
        Normalization stats with keys: feat_mean, feat_std, targ_mean, targ_std.
    device : torch.device
        CPU or CUDA device.
    num_points : int
        Number of boundary sample points (must match training, default=150).
    use_postproc : bool
        Whether to apply geometric regularization.
    geometry_cfg : dict or None
        Dataset-specific geometry config (AHN or Vaihingen).
        If None, defaults to AHN config.
    reg_cfg_override : dict or None
        Override regularizer config. If None, uses the default REG_CFG.

    Returns
    -------
    shapely.geometry.Polygon or None
    """
    if geometry_cfg is None:
        geometry_cfg = AHN_GEOMETRY_CFG

    reg_params = reg_cfg_override or geometry_cfg.get("reg_cfg", REG_CFG)

    # Project to 2D
    raw_2d = raw_pts[:, :2].astype(np.float64)
    ctr = raw_2d.mean(axis=0)
    raw_norm = raw_2d - ctr

    # 1. Seed (Alpha Shape) — uses dataset-calibrated alpha
    alpha_val, spacing = estimate_alpha(
        raw_norm,
        k=geometry_cfg.get("alpha_k", 8),
        alpha_factor=geometry_cfg.get("alpha_factor", 0.82),
        return_spacing=True,
    )
    hull = alpha_shape(raw_norm, alpha_val)
    if hull is None:
        return None

    hp = sample_hull(hull, num_points)

    if model is None or stats is None:
        # Fallback to seed-only (with optional snap)
        if use_postproc:
            snapped = geometric_regularizer(hp, **reg_params)
            final_poly = Polygon(snapped + ctr)
        else:
            final_poly = Polygon(hp + ctr)
        if not final_poly.is_valid:
            final_poly = final_poly.buffer(0)
        return final_poly if not final_poly.is_empty else None

    # 2. Extract Features (4 channels matching training)
    phi = compute_tangent_angles(hp)
    feats = build_feature_matrix_srsp(hp, raw_norm, geometry_cfg=geometry_cfg)

    # 3. Normalize Features
    fe_n = (feats - stats["feat_mean"]) / stats["feat_std"]

    # Pad or slice to num_points exactly
    L = min(len(fe_n), num_points)
    in_ch = feats.shape[1]  # should be 4
    xt = np.zeros((in_ch, num_points), dtype=np.float32)
    xt[:, :L] = fe_n[:L].T

    # 4. CNN Inference
    xt_t = torch.from_numpy(xt).unsqueeze(0).to(device)
    pred_shifts_norm = model(xt_t)[0].cpu().numpy().T

    # 5. Denormalize Targets
    sh_loc = pred_shifts_norm[:L] * stats["targ_std"] + stats["targ_mean"]

    # 6. Local to Global
    sh_glob = local_to_global(sh_loc, phi[:L])
    refined = hp[:L] + sh_glob

    # 7. Geometric Snap
    snapped = geometric_regularizer(refined, **reg_params) if use_postproc else refined

    # Un-normalize center
    final_pts = snapped + ctr

    # Return Shapely Polygon
    try:
        poly = Polygon(final_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILDING INSTANCE ISOLATION (DBSCAN)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_buildings(building_points, eps=1.0, min_points=50):
    """
    Cluster pre-filtered building points into individual building instances
    using DBSCAN.

    Parameters
    ----------
    building_points : np.ndarray
        Nx3 array of points already classified as buildings.
    eps : float
        DBSCAN neighbourhood radius.
    min_points : int
        Minimum cluster size.

    Returns
    -------
    list of np.ndarray
        List of per-building point clouds.
    """
    if len(building_points) == 0:
        return []

    pts_3d = building_points[:, :3] if building_points.shape[1] >= 3 else \
             np.hstack([building_points, np.zeros((len(building_points), 1))])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)

    labels_dbscan = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    clusters = []
    if len(labels_dbscan) == 0 or labels_dbscan.max() < 0:
        return clusters

    for i in range(labels_dbscan.max() + 1):
        cluster_mask = (labels_dbscan == i)
        cluster_pts = building_points[cluster_mask]
        if len(cluster_pts) >= min_points:
            clusters.append(cluster_pts)

    return clusters
