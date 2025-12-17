# TMVANet-SlowFast for Radar Semantic Segmentation

This repository contains the implementation of TMVANet-SlowFast, a deep learning model designed for semantic segmentation of radar imagery. The model leverages a dual-stream architecture (spatial and temporal) to effectively process both the fine-grained details within a single frame and the dynamic evolution across multiple frames.

## Model Architecture Highlights

The core of this project is the `TMVANet_SlowFast_U.py` script, which defines a novel architecture with several key innovations for radar data processing:

1.  **Dual-Stream (Slow-Fast) Design**:
    *   **Spatial Stream (Slow Path)**: Processes a single, high-resolution radar frame to capture rich spatial details, such as the shape and texture of clutter and interference.
    *   **Temporal Stream (Fast Path)**: Processes a sequence of low-resolution frames to model the motion and temporal dynamics of radar echoes.

2.  **GCN-based Temporal Modeling**:
    *   Instead of traditional 3D convolutions, the temporal stream utilizes a Graph Convolutional Network (GCN) to explicitly model the relationships between different frames in the time sequence.
    *   Supports both a **static** (fully-connected) and a **dynamic** (similarity-based) adjacency matrix for ablation studies.

3.  **Spatial Position Enhancement (SPE)**:
    *   A novel module that injects learnable positional encodings into the spatial stream. This allows the model to leverage the inherent positional priors in radar data (e.g., signals near the center vs. at the edge have different characteristics).

4.  **Temporal Attention Aggregation (TTA)**:
    *   An advanced attention mechanism that intelligently fuses features from the time sequence. It calculates attention weights based on both **global** (whole-frame) and **local** (pixel-wise) context, allowing the model to dynamically focus on the most informative frames and regions.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd T-RODNet-main
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n trodnet python=3.8
    conda activate trodnet
    ```

3.  **Install dependencies:**
    This project relies on PyTorch and several other libraries. Install them using pip:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install tqdm matplotlib numpy opencv-python tensorboard
    ```
    *Note: Adjust the PyTorch command based on your CUDA version.*

## Data Preparation

The model is trained on `.npy` files. Please structure your dataset as follows:

```
dataset_npy/
├── train/
│   ├── sample_00001_data.npy
│   ├── sample_00001_label.npy
│   └── ...
└── test/
    ├── sample_00100_data.npy
    ├── sample_00100_label.npy
    └── ...

dataset_npy_2d/
├── train/
│   ├── ... (2D data)
└── test/
    ├── ... (2D data)
```
-   **`_data.npy`**: Should contain the radar data tensor. For temporal/dual modes, the expected shape is `(T, H, W)`, where T is the number of time frames. For spatial mode, the shape should be `(H, W)`.
-   **`_label.npy`**: Should contain the corresponding ground truth segmentation map with shape `(H, W)`.

## How to Use the Model

The main script for training and testing is `train_tmvanet_slowfast.py`. It provides a flexible command-line interface to control the model's behavior.

### Training

**Standard Training (Dual mode with GCN and SPE):**
```bash
python train_tmvanet_slowfast.py --mode train --data-root dataset_npy --batch-size 4 --gpu-ids 0 --run-name "TMVANet_dual_gcn_spe"
```

**Resume Training from a Checkpoint:**
```bash
python train_tmvanet_slowfast.py --mode train --data-root dataset_npy --resume --run-name "TMVANet_resume_run"
```

### Testing and Visualization

This will run the model on the test set and save the visualization results (input, ground truth, prediction) as separate images in the specified results directory.

```bash
python train_tmvanet_slowfast.py --mode test --weights "path/to/your/best.pth" --data-root dataset_npy --results-dir "results/test_run_1"
```

### Ablation Studies (Examples)

The script is designed to facilitate ablation studies through its command-line arguments.

**1. Spatial vs. Temporal vs. Dual Mode:**
- **Spatial Only**: `--model-mode spatial` (will automatically use `dataset_npy_2d`)
- **Temporal Only**: `--model-mode temporal`
- **Dual (Default)**: `--model-mode dual`

```bash
# Example: Train spatial stream only
python train_tmvanet_slowfast.py --mode train --model-mode spatial --run-name "ablation_spatial_only"
```

**2. GCN vs. 3D Convolution:**
- **GCN (Default)**: `--temporal-mode gcn`
- **3D Conv**: `--temporal-mode 3dconv`

```bash
# Example: Train with 3D Convolutions instead of GCN
python train_tmvanet_slowfast.py --mode train --temporal-mode 3dconv --run-name "ablation_3dconv"
```

**3. Number of GCN Layers:**
- Use `--num-gcn-blocks` to stack multiple GCN blocks.

```bash
# Example: Train with 3 GCN layers
python train_tmvanet_slowfast.py --mode train --num-gcn-blocks 3 --run-name "ablation_3_gcn_layers"
```

**4. Static vs. Dynamic Adjacency Matrix in GCN:**
- **Static (Default)**: `--adj-matrix-mode static`
- **Dynamic**: `--adj-matrix-mode dynamic`

```bash
# Example: Train with a dynamic, similarity-based graph
python train_tmvanet_slowfast.py --mode train --adj-matrix-mode dynamic --run-name "ablation_dynamic_adj"
```

**5. Disabling Spatial Position Enhancement (SPE):**
- To turn SPE off, add the `--no-spe` flag.

```bash
# Example: Train without the SPE module
python train_tmvanet_slowfast.py --mode train --no-spe --run-name "ablation_no_spe"
```

## Visualization Tools

Two utility scripts are provided to help you inspect your data.

### 1. Visualizing Raw JSON Labels (`visualize_raw_gt.py`)

This script is for visualizing your *original* dataset (before preprocessing into `.npy` files). It reads `.png` images and their corresponding `.json` label files.

**Usage:**
```bash
python visualize_raw_gt.py --image-dir "path/to/png/images" --label-dir "path/to/json/labels" --output-dir "output/raw_viz"
```
*Note: You may need to install OpenCV (`pip install opencv-python`).*

### 2. Visualizing Preprocessed `.npy` Data (`visualize_gt.py`)

This script visualizes the ground truth from your preprocessed `.npy` dataset, allowing you to verify that the data is correctly formatted before training.

**Usage:**
```bash
# Visualize data from the default test set
python visualize_gt.py

# Visualize data from a different directory
python visualize_gt.py --data-dir dataset_npy_2d/train --output-dir "output/npy_viz"
```

