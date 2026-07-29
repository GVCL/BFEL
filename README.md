# BFEL — Building Footprint Extraction from 3D Airborne LiDAR Point Clouds

This repository contains the code behind the **Seed–Refine–Snap** workflow for extracting GIS-ready building footprints from airborne LiDAR point clouds. The pipeline combines:

1. **RandLA-Net** for semantic segmentation,
2. **DBSCAN** for building instance isolation,
3. **Adaptive alpha-shape seed generation** for initial footprint boundaries,
4. **GeoRefineNet** for learned boundary refinement, and
5. a **conservative geometric regularizer** that snaps the refined boundary into a cleaner orthogonal polygon.

The goal is to turn raw `.las` / `.laz` / `.ply` point clouds into usable building footprint polygons with minimal manual cleanup.

## What this project does

- extracts building points from raw LiDAR,
- separates individual buildings,
- builds an initial polygon seed from the building points,
- refines the boundary with a lightweight 1D CNN,
- regularizes the result into a cleaner orthogonal footprint,
- exports GIS-ready polygons and optional 3D footprint representations.

## Methodology

The method follows a three-stage design:

### 1) Seed
A building instance is converted into a 2D alpha-shape hull. The hull is uniformly resampled to a fixed number of boundary points, and each point receives four geometry-aware features:

- `chord_dev`
- `linearity`
- `density`
- `signal_prior`

These are the only input features used by the refinement model.

### 2) Refine
A lightweight circular 1D CNN, **GeoRefineNet**, predicts per-point displacement in the local tangent-normal frame.  
The network is trained to move each seed point toward the ground-truth boundary while preserving closed-loop polygon structure.

### 3) Snap
A rule-based geometric regularizer cleans the refined boundary by:
- removing redundant vertices,
- detecting wall/corner structure,
- aligning near-orthogonal wall segments,
- rejecting snaps that distort area too much.

This stage is conservative by design: it improves structure without overcorrecting the footprint.

## Repository contents

- `randla_universal.py`  
  RandLA-Net training / validation / testing helper module. Includes memory-safe `.las` / `.laz` loading, dataset wrappers, and evaluation utilities.

- `footprint_inference.py`  
  Inference module for the Seed–Refine–Snap pipeline. Includes feature extraction, GeoRefineNet definition, alpha-shape logic, boundary regularisation, and single-building footprint extraction.

- `seed_refine_snap_v10_with_vaihingen_regularizer.ipynb`  
  Main research notebook for preprocessing, training, ablations, evaluation, and cross-dataset testing.

- `Ground_Truth_Creation.ipynb`  
  Creates matched building-level training / evaluation data and exports per-building files.

- `End_to_End_Footprint_Pipeline.ipynb`  
  End-to-end inference notebook from raw LiDAR input to shapefile export.

- `requirements.txt`  
  Python dependencies needed to run the notebooks and modules.

- `BFEL__Building_Footprint_Extraction_from_3D_Airborne_LiDAR_Point_Clouds__KDIR_2026.pdf`  
  Paper describing the method and experiments.

## Data flow

1. Load raw LiDAR tile (`.las` / `.laz` / `.ply`)
2. Run RandLA-Net semantic segmentation
3. Keep building points only
4. Cluster building points with DBSCAN
5. Generate alpha-shape seed boundary
6. Sample the boundary to a fixed length
7. Extract 4 local geometry features
8. Run GeoRefineNet to predict boundary displacement
9. Apply geometric regularization
10. Export polygons to GIS formats

## Requirements

Install the dependencies in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Key packages used by the project:

- `torch`
- `open3d`
- `laspy`
- `geopandas`
- `shapely`
- `scipy`
- `scikit-learn`
- `numpy`
- `matplotlib`

## Quick start

### 1. Train / prepare the footprint model
Run the preprocessing and training notebook:

```text
seed_refine_snap_v10_with_vaihingen_regularizer.ipynb
```

This notebook handles:

- building-level preprocessing,
- feature and target normalisation,
- training the refinement CNN,
- validation,
- ablation studies,
- cross-dataset testing on Vaihingen.

### 2. Run end-to-end inference
Open:

```text
End_to_End_Footprint_Pipeline.ipynb
```

Set the input and output paths, then run the cells. The pipeline will:

- tile the raw point cloud if needed,
- run semantic segmentation,
- isolate building clusters,
- extract and refine each footprint,
- export the results as shapefiles.

### 3. Use the inference module directly
For programmatic use, import `footprint_inference.py` and call:

- `cluster_buildings(...)`
- `extract_single_footprint(...)`
- `geometric_regularizer(...)`

These are the main entry points for building-level footprint extraction.

## Training notes

The training notebook uses cached building-level samples and normalises features and displacement targets using statistics computed from the training split.  
It supports:

- feature ablations,
- component ablations,
- seed-only baseline evaluation,
- refined vs snapped stage comparison,
- cross-dataset evaluation.

## Inference notes

The inference code is configured for two dataset regimes:

- **AHN**
- **Vaihingen**

Each dataset has its own geometry settings for spacing, alpha-shape scaling, feature radii, and regularisation parameters. The code keeps these settings separate so the pipeline stays stable across point-density differences.

## Outputs

Depending on the notebook and export step, the pipeline can produce:

- `.shp` footprint files
- `.gpkg` files
- `.las` exports per building
- cached `.pkl` samples for training and evaluation
- preview plots for qualitative inspection

## Reproducibility

For best results:

- keep the training and inference geometry settings aligned,
- use the same `point_count` for sampling the boundary,
- reuse the saved normalisation statistics with the trained checkpoint,
- make sure the label mapping for building / non-building matches your dataset.

## Citation

If you use this code in your work, please cite the BFEL paper included in this repository.
