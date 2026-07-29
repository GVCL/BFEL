"""
Universal RandLA-Net Training / Validation / Testing helper module.
Import this from the notebook.  All heavy lifting lives here so the
notebook cells stay short.

Memory-safe: LAZ/LAS files are read via chunked streaming (laspy.open +
chunk_iterator) with on-the-fly voxel sub-sampling so that multi-GB files
never fully reside in RAM.

Enhancements applied on import:
  - Open3D Cache double-read bug patched (Address upstream latency
    ).
  - Default num_workers=4 so CPU KNN (transform) runs in background workers,
    enabling asynchronous data loading..
  - pin_memory=True by default for faster CPU→GPU transfers.
  - persistent_workers=True to avoid worker re-spawn overhead per epoch.
"""

import os, glob, gc, time, logging, warnings
from pathlib import Path
from os.path import join, exists
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.neighbors import KDTree

import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
from open3d._ml3d.torch.dataloaders import (
    get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher,
)
from open3d._ml3d.torch.modules.losses import SemSegLoss, filter_valid_label
from open3d._ml3d.torch.modules.metrics import SemSegMetric
from open3d._ml3d.utils import make_dir

# Set base directory for publication standard
BASE_DIR = "/mnt/d/Saqib/Footprint_Extraction/DIF/Code"


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


try:
    from open3d._ml3d.utils.dataset_helper import Cache as _O3DCache

    def _patched_cache_call(self, unique_id, *data):
        """Patched Cache.__call__ — eliminates redundant disk read on cache hit."""
        fpath = os.path.join(self.cache_dir, f'{unique_id}.npy')
        if not os.path.exists(fpath):
            # Cache miss: compute, write, return (no extra read)
            output = self.func(*data)
            self._write(output, fpath)
            self.cached_ids.append(unique_id)
            return output
        # Cache hit: single read (upstream had a second redundant read here)
        return self._read(fpath)

    _O3DCache.__call__ = _patched_cache_call
    log.info("[perf] Open3D Cache double-read patch applied.")
except Exception as _patch_err:
    log.warning(f"[perf] Could not patch Open3D Cache: {_patch_err}")


try:
    from open3d._ml3d.datasets.samplers.semseg_spatially_regular import (
        SemSegSpatiallyRegularSampler as _SSRSampler,
    )

    _orig_init_dl = _SSRSampler.initialize_with_dataloader

    def _patched_init_dl(self, dataloader):
        _orig_init_dl(self, dataloader)
        # Guarantee cloud_id always exists so get_point_sampler() never raises
        self.cloud_id = 0

    _SSRSampler.initialize_with_dataloader = _patched_init_dl
    log.info("[perf] SemSegSpatiallyRegularSampler cloud_id patch applied.")
except Exception as _patch_err2:
    log.warning(f"[perf] Could not patch SemSegSpatiallyRegularSampler: {_patch_err2}")



# ────────────────────────────────────────────────────────────────────
# 0.  Memory-safe helpers
# ────────────────────────────────────────────────────────────────────

def _fmt_mem(nbytes):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def get_memory_usage():
    """Return a formatted string with current process RSS memory."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        return _fmt_mem(rss)
    except ImportError:
        return "N/A (install psutil)"


def _voxel_subsample(points, labels=None, features=None, grid_size=0.06):
    """
    Pure-numpy voxel grid sub-sampling (matches Open3D DataProcessing logic).
    For each voxel cell, keeps the centroid point and majority-vote label.

    Args:
        points:   (N, 3) float32
        labels:   (N,)   int32  or None
        features: (N, d) float32 or None
        grid_size: voxel edge length in meters

    Returns:
        sub_points, sub_labels, sub_features  (any can be None if input was None)
    """
    if points.shape[0] == 0:
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_lbl = np.zeros((0,), dtype=np.int32) if labels is not None else None
        empty_feat = np.zeros((0, features.shape[1]), dtype=np.float32) if features is not None else None
        return empty_pts, empty_lbl, empty_feat

    # Compute voxel keys
    voxel_idx = np.floor(points / grid_size).astype(np.int64)
    # Pack (ix, iy, iz) into a single int64 key for fast grouping
    # Shift to positive range first
    voxel_idx -= voxel_idx.min(axis=0)
    dims = voxel_idx.max(axis=0) + 1
    keys = (voxel_idx[:, 0] * dims[1] * dims[2] +
            voxel_idx[:, 1] * dims[2] +
            voxel_idx[:, 2])

    # Get unique voxels and group indices
    if labels is not None:
        unique_keys, first_idx, inverse, counts = np.unique(
            keys, return_index=True, return_inverse=True, return_counts=True
        )
        sub_labels = labels[first_idx]
    else:
        unique_keys, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        sub_labels = None

    n_voxels = len(unique_keys)

    # Compute centroids via scatter-add
    sub_points = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sub_points, inverse, points)
    sub_points /= counts[:, None]
    sub_points = sub_points.astype(np.float32)

    # Average features
    sub_features = None
    if features is not None:
        sub_features = np.zeros((n_voxels, features.shape[1]), dtype=np.float64)
        np.add.at(sub_features, inverse, features)
        sub_features /= counts[:, None]
        sub_features = sub_features.astype(np.float32)

    return sub_points, sub_labels, sub_features


def _read_laz_chunked(pc_path, grid_size=0.06, remap_dict=None,
                      label_field="classification", feat_fields=None,
                      chunk_size=500_000):
    """
    Memory-safe LAZ/LAS reader using laspy.open() + chunk_iterator().

    Instead of loading the entire file into RAM, reads in chunks and
    voxel-subsamples each chunk on the fly. Final result is a merged,
    sub-sampled point cloud that fits comfortably in memory.

    Args:
        pc_path:      Path to .laz/.las file
        grid_size:    Voxel edge size for sub-sampling (meters)
        remap_dict:   {new_label: [old_label_list]} or None
        label_field:  Name of the label dimension in the LAS file
        feat_fields:  List of extra feature field names or None
        chunk_size:   Number of points per chunk (default 500K)

    Returns:
        dict with 'point' (N,3), 'label' (N,), 'feat' (N,d) or None
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("Please run: pip install laspy[lazrs]")

    if feat_fields is None:
        feat_fields = []

    basename = os.path.basename(pc_path)

    # Fast direct full read if the file is small (e.g. pre-tiled datasets)
    with laspy.open(pc_path) as reader:
        total = reader.header.point_count

    if total <= chunk_size:
        las_data = laspy.read(pc_path)
        n = len(las_data)

        # --- Extract XYZ ---
        pts = np.zeros((n, 3), dtype=np.float32)
        pts[:, 0] = las_data.x
        pts[:, 1] = las_data.y
        pts[:, 2] = las_data.z

        # --- Extract labels ---
        try:
            if hasattr(las_data, label_field):
                lbl = np.array(getattr(las_data, label_field), dtype=np.int32)
            elif hasattr(las_data, "classification"):
                lbl = np.array(las_data.classification, dtype=np.int32)
            else:
                lbl = np.zeros(n, dtype=np.int32)
        except Exception:
            lbl = np.zeros(n, dtype=np.int32)
        lbl = lbl.reshape(-1)

        # --- Remap labels ---
        if remap_dict is not None and len(remap_dict) > 0:
            new_lbl = np.zeros_like(lbl)
            for new_c, old_c_list in remap_dict.items():
                new_lbl[np.isin(lbl, old_c_list)] = new_c
            lbl = new_lbl

        # --- Extract features ---
        feat = None
        if feat_fields:
            feats_list = []
            for ff in feat_fields:
                if hasattr(las_data, ff):
                    feats_list.append(
                        np.array(getattr(las_data, ff), dtype=np.float32))
            if feats_list:
                feat = (np.vstack(feats_list).T if len(feats_list) > 1
                        else feats_list[0].reshape(-1, 1)
                        if feats_list[0].ndim == 1
                        else feats_list[0])

        # --- Voxel sub-sample ---
        points, labels, features = _voxel_subsample(
            pts, lbl, feat, grid_size=grid_size)

        return {"point": points, "feat": features, "label": labels}

    # Otherwise, fall back to memory-safe chunked reading for massive files
    all_points = []
    all_labels = []
    all_feats  = []

    with laspy.open(pc_path) as reader:
        # Disable inner progress bar for individual chunk reading to avoid log spam
        pbar = tqdm(total=total, desc=f"  ↳ Reading {basename}", unit=" pts",
                    unit_scale=True, leave=False, disable=True)

        for chunk in reader.chunk_iterator(chunk_size):
            n = len(chunk)

            # --- Extract XYZ ---
            pts = np.zeros((n, 3), dtype=np.float32)
            pts[:, 0] = chunk.x
            pts[:, 1] = chunk.y
            pts[:, 2] = chunk.z

            # --- Extract labels ---
            try:
                if hasattr(chunk, label_field):
                    lbl = np.array(getattr(chunk, label_field), dtype=np.int32)
                elif hasattr(chunk, "classification"):
                    lbl = np.array(chunk.classification, dtype=np.int32)
                else:
                    lbl = np.zeros(n, dtype=np.int32)
            except Exception:
                lbl = np.zeros(n, dtype=np.int32)
            lbl = lbl.reshape(-1)

            # --- Remap labels ---
            if remap_dict is not None and len(remap_dict) > 0:
                new_lbl = np.zeros_like(lbl)
                for new_c, old_c_list in remap_dict.items():
                    new_lbl[np.isin(lbl, old_c_list)] = new_c
                lbl = new_lbl

            # --- Extract features ---
            feat = None
            if feat_fields:
                feats_list = []
                for ff in feat_fields:
                    if hasattr(chunk, ff):
                        feats_list.append(
                            np.array(getattr(chunk, ff), dtype=np.float32))
                if feats_list:
                    feat = (np.vstack(feats_list).T if len(feats_list) > 1
                            else feats_list[0].reshape(-1, 1)
                            if feats_list[0].ndim == 1
                            else feats_list[0])

            # --- Voxel sub-sample this chunk ---
            sub_pts, sub_lbl, sub_feat = _voxel_subsample(
                pts, lbl, feat, grid_size=grid_size)

            all_points.append(sub_pts)
            all_labels.append(sub_lbl)
            if sub_feat is not None:
                all_feats.append(sub_feat)

            pbar.update(n)

            # Free chunk memory
            del pts, lbl, feat, chunk
            gc.collect()

        pbar.close()

    # Merge all sub-sampled chunks
    # print(f"  ↳ Merging {len(all_points)} sub-sampled chunks...")
    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    features = np.concatenate(all_feats, axis=0) if all_feats else None

    del all_points, all_labels, all_feats
    gc.collect()

    # Final global sub-sample to remove cross-chunk duplicates
    n_before = points.shape[0]
    # print(f"  ↳ Final voxel sub-sample ({n_before:,} → ", end="", flush=True)
    points, labels, features = _voxel_subsample(
        points, labels, features, grid_size=grid_size)
    # print(f"{points.shape[0]:,} points, grid={grid_size}m)")

    gc.collect()
    return {"point": points, "feat": features, "label": labels}


# ────────────────────────────────────────────────────────────────────
# 1.  Universal Split / Dataset
# ────────────────────────────────────────────────────────────────────

def _resolve_files(path_or_glob):
    """Accept a list of paths, a single file, a directory, or a glob and return list of paths."""
    # Already a list of paths — return as-is (convert Path objects to str)
    if isinstance(path_or_glob, (list, tuple)):
        return [str(p) for p in path_or_glob]
    if os.path.isfile(path_or_glob):
        return [path_or_glob]
    if os.path.isdir(path_or_glob):
        files = []
        for ext in ("*.ply", "*.laz", "*.las"):
            files.extend(glob.glob(join(path_or_glob, ext)))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No .ply or .laz files in {path_or_glob}")
        return files
    files = sorted(glob.glob(path_or_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {path_or_glob}")
    return files


class UniversalSplit(BaseDatasetSplit):
    """A dataset split that reads PLY files with configurable field names."""

    def __init__(self, dataset, split="training"):
        # We must call super().__init__ which sets up self.sampler
        super().__init__(dataset, split=split)
        log.info(f"Found {len(self.path_list)} pointclouds for {split}")

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        cfg = self.dataset.cfg
        point_field = cfg.get("point_field", "positions")
        label_field = cfg.get("label_field", "class")
        feat_fields = cfg.get("feat_fields", [])
        grid_size = cfg.get("grid_size_read", 0.06)
        chunk_size = cfg.get("laz_chunk_size", 500_000)

        ext = pc_path.lower().split('.')[-1]
        basename = os.path.basename(pc_path)
        # print(f"  Loading: {basename} (format={ext})")
        
        if ext in ("ply", "pcd"):
            # print(f"  ↳ Reading PLY via Open3D ...")
            pc = o3d.t.io.read_point_cloud(pc_path).point
            points = pc[point_field].numpy().astype(np.float32)

            try:
                labels = pc[label_field].numpy().astype(np.int32).reshape((-1,))
            except Exception:
                labels = np.zeros(points.shape[0], dtype=np.int32)

            remap_dict = getattr(cfg, 'remap_classes', None)
            if remap_dict is not None and len(remap_dict) > 0:
                new_labels = np.zeros_like(labels)
                for new_c, old_c_list in remap_dict.items():
                    new_labels[np.isin(labels, old_c_list)] = new_c
                labels = new_labels

            feat = None
            if feat_fields:
                feats = []
                for ff in feat_fields:
                    if ff in pc:
                        feats.append(pc[ff].numpy().astype(np.float32))
                if feats:
                    feat = np.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]
                    if feat.ndim == 1:
                        feat = feat.reshape(-1, 1)

            # print(f"  ↳ Loaded {points.shape[0]:,} points  [RAM: {get_memory_usage()}]")
            return {"point": points, "feat": feat, "label": labels}

        elif ext in ("laz", "las"):
            # ── Memory-safe chunked reader (never loads full file) ──
            remap_dict = getattr(cfg, 'remap_classes', None)
            result = _read_laz_chunked(
                pc_path,
                grid_size=grid_size,
                remap_dict=remap_dict,
                label_field=label_field,
                feat_fields=feat_fields,
                chunk_size=chunk_size,
            )
            # print(f"  ↳ Final: {result['point'].shape[0]:,} points  [RAM: {get_memory_usage()}]")
            return result
                    
        else:
            raise ValueError(f"Unsupported file format: {pc_path}")

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.stem
        return {"idx": idx, "name": name, "path": str(pc_path), "split": self.split}


class UniversalPointCloudDataset(BaseDataset):
    """Works with any PLY point cloud dataset.  No YAML config needed."""

    def __init__(
        self,
        train_files,
        val_files,
        test_files,
        name="UniversalPC",
        num_classes=6,
        ignored_label_inds=None,
        label_to_names=None,
        class_weights=None,
        cache_dir="./logs/cache",
        use_cache=True,
        num_points=65536,
        test_result_folder="./test",
        point_field="positions",
        label_field="class",
        feat_fields=None,
        steps_per_epoch_train=100,
        steps_per_epoch_valid=10,
        sampler=None,
        grid_size_read=0.06,
        laz_chunk_size=500_000,
        **kwargs,
    ):
        if ignored_label_inds is None:
            ignored_label_inds = [0]
        if label_to_names is None:
            label_to_names = {i: f"class_{i}" for i in range(num_classes)}
        if class_weights is None:
            class_weights = []
        if feat_fields is None:
            feat_fields = []
        if sampler is None:
            sampler = {"name": "SemSegSpatiallyRegularSampler"}

        self._train_files = _resolve_files(train_files)
        self._val_files   = _resolve_files(val_files)
        self._test_files  = _resolve_files(test_files)

        super().__init__(
            dataset_path=os.path.dirname(self._train_files[0]),
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            num_points=num_points,
            test_result_folder=test_result_folder,
            ignored_label_inds=ignored_label_inds,
            class_weights=class_weights,
            point_field=point_field,
            label_field=label_field,
            feat_fields=feat_fields,
            steps_per_epoch_train=steps_per_epoch_train,
            steps_per_epoch_valid=steps_per_epoch_valid,
            sampler=sampler,
            grid_size_read=grid_size_read,
            laz_chunk_size=laz_chunk_size,
            **kwargs,
        )
        self._label_to_names = label_to_names
        self.num_classes = num_classes
        self.label_values = np.sort(list(label_to_names.keys()))
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(ignored_label_inds)

    @staticmethod
    def get_label_to_names():
        # Will be overridden per-instance
        return {}

    def get_label_to_names(self):
        return self._label_to_names

    def get_split(self, split):
        return UniversalSplit(self, split=split)

    def get_split_list(self, split):
        if split in ("test", "testing"):
            return list(self._test_files)
        elif split in ("val", "validation"):
            return list(self._val_files)
        elif split in ("train", "training"):
            return list(self._train_files)
        elif split == "all":
            return list(self._train_files + self._val_files + self._test_files)
        raise ValueError(f"Invalid split {split}")

    def is_tested(self, attr):
        name = attr["name"]
        store = join(self.cfg.test_result_folder, self.name, name + ".txt")
        return exists(store)

    def save_test_result(self, results, attr):
        name = attr["name"].split(".")[0]
        path = self.cfg.test_result_folder
        make_dir(path)
        pred = results["predict_labels"]
        store = join(path, self.name, name + ".txt")
        make_dir(Path(store).parent)
        np.savetxt(store, pred.astype(np.int32), fmt="%d")
        log.info(f"Saved {name} in {store}")


# ────────────────────────────────────────────────────────────────────
# 2.  Extended metrics (confusion-matrix based)
# ────────────────────────────────────────────────────────────────────

def compute_extended_metrics(confusion_matrix, label_to_names=None, ignored_label_inds=None):
    """
    From a C×C confusion matrix compute:
      mIoU, OA, mAcc, per-class F1, avg-F1
    Returns a dict with everything.
    """
    if ignored_label_inds is None:
        ignored_label_inds = []
    C = confusion_matrix.shape[0]
    cm = confusion_matrix.astype(np.float64)

    per_class = {}
    ious, accs, f1s = [], [], []

    for c in range(C):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        acc = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = acc  # recall == per-class accuracy
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

        name = label_to_names.get(c, f"class_{c}") if label_to_names else f"class_{c}"
        per_class[c] = {"name": name, "iou": iou, "acc": acc, "prec": prec, "f1": f1}

        if c not in ignored_label_inds:
            ious.append(iou)
            accs.append(acc)
            f1s.append(f1)

    # Overall accuracy
    total_correct = np.trace(cm)
    total_pts = cm.sum()
    oa = total_correct / total_pts if total_pts > 0 else 0.0

    return {
        "mIoU": float(np.nanmean(ious)),
        "OA": float(oa),
        "mAcc": float(np.nanmean(accs)),
        "avg_F1": float(np.nanmean(f1s)),
        "per_class": per_class,
    }


def print_metrics(metrics, title="Evaluation"):
    """Pretty-print a metrics dict."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  mIoU  : {metrics['mIoU']*100:.2f}%")
    print(f"  OA    : {metrics['OA']*100:.2f}%")
    print(f"  mAcc  : {metrics['mAcc']*100:.2f}%")
    print(f"  Avg F1: {metrics['avg_F1']*100:.2f}%")
    print(f"{'─'*70}")
    print(f"  {'Class':<30s} {'IoU':>8s} {'Acc':>8s} {'Prec':>8s} {'F1':>8s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for c, info in metrics["per_class"].items():
        print(
            f"  {info['name']:<30s}"
            f" {info['iou']*100:7.2f}%"
            f" {info['acc']*100:7.2f}%"
            f" {info['prec']*100:7.2f}%"
            f" {info['f1']*100:7.2f}%"
        )
    print(f"{'='*70}\n")


# ────────────────────────────────────────────────────────────────────
# 3.  Custom training loop
# ────────────────────────────────────────────────────────────────────

def run_training(
    model,
    dataset,
    device,
    max_epochs=100,
    batch_size=2,
    val_batch_size=2,
    lr=0.001,
    scheduler_gamma=0.9886,
    validate_every=5,
    save_ckpt_freq=5,
    num_workers=0,        # SemSegSpatiallyRegularSampler is NOT multiprocessing-safe;
                          # cloud_id is shared state set only in main-process iteration.
    pin_memory=True,      # faster CPU→GPU pinned-memory transfers
    persistent_workers=False,  # requires num_workers > 0
    log_dir="./logs",
    train_sum_dir="train_log",
    ckpt_path=None,
):
    """
    Custom training loop with periodic validation and extended metrics.
    Returns path to the best checkpoint (by mIoU).

    Performance notes:
      - num_workers > 0 moves CPU KNN (transform) to background processes so
        the GPU is not blocked waiting for data preparation.
      - pin_memory=True allocates host tensors in pinned memory for faster
        CPU→GPU transfers via DMA.
      - persistent_workers=True keeps worker processes alive between epochs
        to avoid per-epoch fork/join overhead.
      - For WSL users: ensure cache_dir is on the Linux filesystem (e.g.
        /home/<user>/...) not on /mnt/d/ (NTFS), which has ~5-10x slower
        small random-read throughput.
    """
    # Clamp num_workers: multiprocessing can deadlock on some WSL setups
    # if cache is on NTFS (/mnt/d). Detect and warn.
    _cache_dir = getattr(dataset.cfg, 'cache_dir', '')
    _on_ntfs   = str(_cache_dir).startswith('/mnt/')
    if num_workers > 0 and _on_ntfs:
        print(f"  ⚠️  [PERF] cache_dir={_cache_dir!r} is on NTFS/WSL bridge.")
        print(f"  ⚠️  [PERF] Recommend moving cache to Linux fs (e.g. /home/<user>/randlanet_cache)")
        print(f"  ⚠️  [PERF] Setting num_workers=0 temporarily to avoid WSL multiprocessing issues.")
        num_workers = 0
        persistent_workers = False

    # persistent_workers requires num_workers > 0
    if num_workers == 0:
        persistent_workers = False

    print(f"\n{'─'*70}")
    print(f"  [SETUP] Moving model to {device} ...")
    model.to(device)
    model.device = device
    if device.type == 'cuda':
        print(f"  [SETUP] GPU: {torch.cuda.get_device_name(0)}")
        print(f"  [SETUP] GPU Memory: {torch.cuda.memory_allocated()/(1024**2):.0f}MB allocated, "
              f"{torch.cuda.memory_reserved()/(1024**2):.0f}MB reserved")
    print(f"  [SETUP] Process RAM: {get_memory_usage()}")
    print(f"  [SETUP] num_workers={num_workers}  pin_memory={pin_memory}  "
          f"persistent_workers={persistent_workers}")

    logs_dir = join(
        log_dir,
        "RandLANet_" + dataset.name + "_torch",
    )
    ckpt_dir = join(logs_dir, "checkpoint")
    make_dir(ckpt_dir)
    print(f"  [SETUP] Checkpoint dir: {ckpt_dir}")

    # ── Batcher ──
    batcher_name = getattr(model.cfg, "batcher", "DefaultBatcher")
    batcher = DefaultBatcher() if batcher_name == "DefaultBatcher" else ConcatBatcher(device, model.cfg.name)

    # ── Train loader ──
    print(f"\n  [DATALOADER] Building TRAIN dataloader ...")
    print(f"  [DATALOADER]   Files: {len(dataset.get_split_list('train'))}, "
          f"batch_size={batch_size}, cache={dataset.cfg.use_cache}")
    t0 = time.time()
    train_dataset = dataset.get_split("train")
    train_sampler = train_dataset.sampler
    train_split = TorchDataloader(
        dataset=train_dataset,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=train_sampler,
        use_cache=dataset.cfg.use_cache,
        steps_per_epoch=dataset.cfg.get("steps_per_epoch_train", None),
    )
    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        sampler=get_sampler(train_sampler),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=batcher.collate_fn,
        worker_init_fn=lambda x: np.random.seed(
            x + np.uint32(torch.utils.data.get_worker_info().seed)
        ) if num_workers > 0 else None,
    )
    print(f"  [DATALOADER] Train dataloader ready ({time.time()-t0:.1f}s)  [RAM: {get_memory_usage()}]")

    # ── Val loader ──
    print(f"\n  [DATALOADER] Building VALIDATION dataloader ...")
    t0 = time.time()
    valid_dataset = dataset.get_split("validation")
    valid_sampler = valid_dataset.sampler
    valid_split = TorchDataloader(
        dataset=valid_dataset,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=valid_sampler,
        use_cache=dataset.cfg.use_cache,
        steps_per_epoch=dataset.cfg.get("steps_per_epoch_valid", None),
    )
    valid_loader = DataLoader(
        valid_split,
        batch_size=val_batch_size,
        sampler=get_sampler(valid_sampler),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=batcher.collate_fn,
        worker_init_fn=lambda x: np.random.seed(
            x + np.uint32(torch.utils.data.get_worker_info().seed)
        ) if num_workers > 0 else None,
    )
    print(f"  [DATALOADER] Val dataloader ready ({time.time()-t0:.1f}s)  [RAM: {get_memory_usage()}]")

    # ── Optimizer / Scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_gamma)
    print(f"\n  [SETUP] Optimizer: Adam(lr={lr}), Scheduler: ExpLR(γ={scheduler_gamma})")

    # ── Loss ──
    Loss = SemSegLoss(None, model, dataset, device)

    # ── Resume ──
    start_epoch = 0
    if ckpt_path and exists(ckpt_path):
        print(f"  [RESUME] Loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        print(f"  [RESUME] Resuming from epoch {start_epoch}")

    best_miou = -1.0
    best_ckpt = None

    label_to_names = dataset.get_label_to_names()
    ignored = list(dataset.cfg.ignored_label_inds) if hasattr(dataset.cfg, "ignored_label_inds") else [0]

    print(f"\n{'='*70}")
    print(f"  TRAINING RandLA-Net  |  epochs {start_epoch}→{max_epochs}  |  val every {validate_every}")
    print(f"  batch_size={batch_size}  |  steps/epoch={len(train_loader)}  |  val_steps={len(valid_loader)}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()

        # ─── Train ───
        model.train()
        metric_train = SemSegMetric()
        losses = []
        model.trans_point_sampler = train_sampler.get_point_sampler()

        # GPU timing accumulators (measures actual GPU compute time vs total)
        _gpu_time = 0.0
        _t_data_start = time.time()

        pbar = tqdm(train_loader, desc=f"[E{epoch:03d}] train",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for step, inputs in enumerate(pbar):
            _data_wait = time.time() - _t_data_start  # time spent waiting for data

            if hasattr(inputs["data"], "to"):
                inputs["data"].to(device)
            optimizer.zero_grad()

            _t_gpu = time.time()
            results = model(inputs["data"])
            loss, gt_labels, predict_scores = model.get_loss(Loss, results, inputs, device)
            if predict_scores.size()[-1] == 0:
                _t_data_start = time.time()
                continue
            loss.backward()
            if hasattr(model.cfg, "grad_clip_norm") and model.cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.grad_clip_norm)
            optimizer.step()
            _gpu_time += time.time() - _t_gpu

            metric_train.update(predict_scores, gt_labels)
            step_loss = loss.cpu().item()
            losses.append(step_loss)
            # Update progress bar with live loss and data-wait indicator
            pbar.set_postfix(
                loss=f"{step_loss:.4f}",
                avg=f"{np.mean(losses):.4f}",
                wait=f"{_data_wait:.2f}s",  # high wait → GPU starved by CPU KNN
            )
            _t_data_start = time.time()

        scheduler.step()
        train_loss = np.mean(losses) if losses else float("nan")
        train_accs = metric_train.acc()
        train_ious = metric_train.iou()
        epoch_time = time.time() - epoch_start

        lr_now = optimizer.param_groups[0]['lr']
        gpu_mem = f"{torch.cuda.memory_allocated()/(1024**2):.0f}MB" if device.type == 'cuda' else 'N/A'
        _gpu_pct = 100.0 * _gpu_time / epoch_time if epoch_time > 0 else 0.0
        print(f"  Epoch {epoch:>4d}  |  loss: {train_loss:.4f}  |  mIoU: {train_ious[-1]*100:.2f}%  "
              f"|  mAcc: {train_accs[-1]*100:.2f}%  |  lr: {lr_now:.6f}  |  {epoch_time:.1f}s  "
              f"|  GPU: {gpu_mem}  |  GPU-busy: {_gpu_pct:.0f}%  |  RAM: {get_memory_usage()}")
        if _gpu_pct < 30 and epoch <= 2:
            print(f"  ⚠️  GPU-busy={_gpu_pct:.0f}% — GPU is starved. "
                  f"Move cache to Linux fs and set num_workers≥4.")

        # ─── Validate ───
        if epoch % validate_every == 0 and epoch > 0:
            print(f"\n  [VALIDATION] Running validation @ epoch {epoch} ...")
            val_start = time.time()
            model.eval()
            metric_val = SemSegMetric()
            valid_losses = []
            model.trans_point_sampler = valid_sampler.get_point_sampler()

            with torch.no_grad():
                for step, inputs in enumerate(tqdm(valid_loader, desc=f"[E{epoch:03d}] val  ")):
                    if hasattr(inputs["data"], "to"):
                        inputs["data"].to(device)
                    results = model(inputs["data"])
                    loss, gt_labels, predict_scores = model.get_loss(Loss, results, inputs, device)
                    if predict_scores.size()[-1] == 0:
                        continue
                    metric_val.update(predict_scores, gt_labels)
                    valid_losses.append(loss.cpu().item())

            val_loss = np.mean(valid_losses) if valid_losses else float("nan")
            val_time = time.time() - val_start
            print(f"  [VALIDATION] val_loss: {val_loss:.4f}  ({val_time:.1f}s)")

            if metric_val.confusion_matrix is not None:
                ext = compute_extended_metrics(metric_val.confusion_matrix, label_to_names, ignored)
                print_metrics(ext, title=f"Validation @ Epoch {epoch}")

                if ext["mIoU"] > best_miou:
                    best_miou = ext["mIoU"]
                    best_ckpt = join(ckpt_dir, "best.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_miou": best_miou,
                        },
                        best_ckpt,
                    )
                    print(f"  ★ New best mIoU: {best_miou*100:.2f}% — saved to {best_ckpt}")
            else:
                print(f"  val loss: {val_loss:.4f} (no valid predictions)")

        # ─── Checkpoint ───
        if epoch % save_ckpt_freq == 0 or epoch == max_epochs:
            ckpt_file = join(ckpt_dir, f"ckpt_{epoch:05d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                ckpt_file,
            )
            print(f"  [CKPT] Saved: {ckpt_file}")

    print(f"\n{'='*70}")
    print(f"  Training complete.  Best mIoU: {best_miou*100:.2f}%")
    print(f"  RAM: {get_memory_usage()}")
    print(f"{'='*70}")
    return best_ckpt or join(ckpt_dir, f"ckpt_{max_epochs:05d}.pth")


# ────────────────────────────────────────────────────────────────────
# 4.  Testing / Evaluation
# ────────────────────────────────────────────────────────────────────

def run_test(
    model,
    dataset,
    device,
    ckpt_path,
    test_batch_size=1,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
):
    """
    Run full inference on the test split, compute extended metrics, and
    return (metrics_dict, list_of_prediction_dicts).
    """
    print(f"\n{'─'*70}")
    print(f"  [TEST] Setting up test pipeline ...")
    model.to(device)
    model.device = device
    model.eval()

    # Load checkpoint
    if ckpt_path and exists(ckpt_path):
        print(f"  [TEST] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [TEST] Checkpoint loaded successfully")
    else:
        print("  [TEST] WARNING: no checkpoint loaded — using current model weights")

    batcher_name = getattr(model.cfg, "batcher", "DefaultBatcher")
    batcher = DefaultBatcher() if batcher_name == "DefaultBatcher" else ConcatBatcher(device, model.cfg.name)

    print(f"  [TEST] Building test dataloader ...")
    t0 = time.time()
    test_dataset = dataset.get_split("test")
    test_sampler = test_dataset.sampler
    test_split = TorchDataloader(
        dataset=test_dataset,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=test_sampler,
        use_cache=dataset.cfg.use_cache,
    )
    # Clamp workers if cache is on NTFS/WSL bridge
    _cache_dir = getattr(dataset.cfg, 'cache_dir', '')
    if num_workers > 0 and str(_cache_dir).startswith('/mnt/'):
        num_workers = 0
        persistent_workers = False
    if num_workers == 0:
        persistent_workers = False

    test_loader = DataLoader(
        test_split,
        batch_size=test_batch_size,
        sampler=get_sampler(test_sampler),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=batcher.collate_fn,
    )
    print(f"  [TEST] Dataloader ready ({time.time()-t0:.1f}s)  [RAM: {get_memory_usage()}]")

    model.trans_point_sampler = test_sampler.get_point_sampler()
    curr_cloud_id = -1
    test_probs = [None] * len(test_dataset)
    all_results = []

    label_to_names = dataset.get_label_to_names()
    ignored = list(dataset.cfg.ignored_label_inds) if hasattr(dataset.cfg, "ignored_label_inds") else [0]

    overall_metric = SemSegMetric()
    test_start = time.time()

    print(f"\n{'='*70}")
    print(f"  TESTING on {len(test_dataset)} tile(s) ...")
    print(f"{'='*70}\n")

    with torch.no_grad():
        for step, inputs in enumerate(tqdm(test_loader, desc="testing")):
            if hasattr(inputs["data"], "to"):
                inputs["data"].to(device)
            results = model(inputs["data"])

            # --- accumulate probs (same logic as pipeline.update_tests) ---
            end_threshold = 0.5
            if curr_cloud_id != test_sampler.cloud_id:
                curr_cloud_id = test_sampler.cloud_id
                if test_probs[curr_cloud_id] is None:
                    num_points = test_sampler.possibilities[curr_cloud_id].shape[0]
                    test_probs[curr_cloud_id] = np.zeros([num_points, model.cfg.num_classes], dtype=np.float16)

            test_probs[curr_cloud_id] = model.update_probs(
                inputs, results, test_probs[curr_cloud_id]
            )

            this_poss = test_sampler.possibilities[test_sampler.cloud_id]
            if (
                this_poss[this_poss > end_threshold].shape[0]
                == this_poss.shape[0]
            ):
                # Cloud complete
                proj_inds = model.preprocess(
                    test_dataset.get_data(curr_cloud_id),
                    {"split": "test"},
                ).get("proj_inds", None)
                if proj_inds is None:
                    proj_inds = np.arange(test_probs[curr_cloud_id].shape[0])

                pred_labels = np.argmax(test_probs[curr_cloud_id][proj_inds], 1)
                pred_scores = test_probs[curr_cloud_id][proj_inds]

                gt_labels = test_dataset.get_data(curr_cloud_id)["label"]

                inference_result = {
                    "predict_labels": pred_labels,
                    "predict_scores": pred_scores,
                    "gt_labels": gt_labels,
                }
                all_results.append(inference_result)

                # Update overall metric
                valid_scores, valid_labels = filter_valid_label(
                    torch.tensor(pred_scores).to(device),
                    torch.tensor(gt_labels).to(device),
                    model.cfg.num_classes,
                    model.cfg.ignored_label_inds,
                    device,
                )
                if valid_labels.size(0) > 0:
                    overall_metric.update(valid_scores, valid_labels)

                attr = test_dataset.get_attr(curr_cloud_id)
                print(f"  ✓ Completed tile: {attr['name']}  "
                      f"({pred_labels.shape[0]:,} pts)  [RAM: {get_memory_usage()}]")

    test_time = time.time() - test_start

    # Compute final metrics
    metrics = None
    if overall_metric.confusion_matrix is not None:
        metrics = compute_extended_metrics(
            overall_metric.confusion_matrix, label_to_names, ignored
        )
        print_metrics(metrics, title="TEST SET EVALUATION")
    else:
        print("No ground truth labels available for evaluation.")

    print(f"  [TEST] Total test time: {test_time:.1f}s  |  RAM: {get_memory_usage()}")
    return metrics, all_results


# ────────────────────────────────────────────────────────────────────
# 5.  Tiling Utility (from tile_dataset.py)
# ────────────────────────────────────────────────────────────────────

def tile_laz(input_file, out_dir, tile_size=50.0):
    """
    Tiles a large LAZ/LAS file into smaller chunks to avoid OOM errors.
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("Please run: pip install laspy[lazrs] to read .laz files")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[TILING] Starting tiling of {os.path.basename(input_file)}")
    print(f"  Tile size: {tile_size}m | Output dir: {out_dir}")
    print(f"  RAM before: {get_memory_usage()}")
    
    # Open the file in chunked mode to avoid loading it all into memory
    with laspy.open(input_file) as file:
        header = file.header
        
        # Dictionary to store tile writers
        writers = {}
        
        # 1 million points per chunk is roughly ~30-40MB of RAM
        chunk_size = 1_000_000 
        
        total_points = header.point_count
        pbar = tqdm(total=total_points, desc=f"Tiling {os.path.basename(input_file)}", unit=" pts", unit_scale=True)
        
        for points in file.chunk_iterator(chunk_size):
            # Calculate grid indices for each point
            x_idx = np.floor((points.x - header.mins[0]) / tile_size).astype(int)
            y_idx = np.floor((points.y - header.mins[1]) / tile_size).astype(int)
            
            # Combine x and y indices to unique tile ids
            tile_ids = np.column_stack((x_idx, y_idx))
            unique_tiles, inverse_indices = np.unique(tile_ids, axis=0, return_inverse=True)
            
            # Distribute points to their respective tiles
            for i, (tx, ty) in enumerate(unique_tiles):
                tile_key = f"{tx}_{ty}"
                
                # Filter points belonging to this tile
                mask = inverse_indices == i
                tile_points = points[mask]
                
                if tile_key not in writers:
                    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_tile_{tile_key}.laz")
                    new_header = laspy.LasHeader(point_format=header.point_format, version=header.version)
                    new_header.offsets = header.offsets
                    new_header.scales = header.scales
                    
                    # Create a writer for this tile
                    writer = laspy.open(out_path, mode="w", header=new_header)
                    writers[tile_key] = writer
                
                # Write the chunk to the correct tile
                writers[tile_key].write_points(tile_points)
                
            pbar.update(len(points))
            
            # Update memory info in progress bar periodically
            pbar.set_postfix(ram=get_memory_usage(), tiles=len(writers))
            
        pbar.close()
        
        # Close all writers
        for writer in writers.values():
            writer.close()
            
    print(f"✓ Tiling complete! Created {len(writers)} tiles.")
    print(f"  RAM after: {get_memory_usage()}\n")
