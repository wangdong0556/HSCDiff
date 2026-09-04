# HSCD: Hierarchical Spatiotemporal Conditional Diffusion

This repository is a clean PyTorch reference implementation of **HSCD** for
video-based 3D human pose estimation. It implements the method described in the
manuscript:

- **MIE** encodes inter-frame motion and intra-frame skeletal structure with
  factorized temporal/spatial attention and gated cross-dimension fusion.
- **CCB** pools the cached MIE representation to every decoder resolution and
  gates it against the current diffusion-state feature.
- **PRD** predicts diffusion noise with a multi-scale temporal U-Net and uses
  the CCB output at every reconstruction scale.
- The forward process uses 1,000 linearly spaced noise levels. Inference uses a
  respaced reverse chain with 50 transitions by default.

The code is independent of D3DP's MixSTE denoiser and multi-hypothesis
aggregation. D3DP was used only as a reference for repository organization and
the expected Human3.6M workflow.

## What is and is not reproduced

The mathematical data flow in the paper is implemented directly. Parameters
that are explicitly reported in the paper are fixed in
`configs/h36m/hscd_b.yaml`. A few low-level engineering choices that are not
specified in the manuscript (for example PRD channel widths and dropout) are
exposed in the YAML file instead of being presented as paper facts. Therefore,
the repository is an executable implementation of the method, but the reported
benchmark numbers require the authors' exact preprocessing, training data, and
undisclosed run-specific choices.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install a CUDA-enabled PyTorch build appropriate for the local driver when
training on an NVIDIA GPU.

## Data format

The loader uses a compact NPZ interchange format so that it is not tied to one
third-party preprocessing repository. Each file contains:

```text
poses_2d: float32 [N, T, 17, 2]
poses_3d: float32 [N, T, 17, 3]
```

2D coordinates should already be normalized to the camera/image convention
used for training. 3D coordinates must be root-relative and stored in meters by
default; the evaluator multiplies errors by 1,000 to report millimeters.
Sequences may be longer than the configured window. Training uses random
windows and evaluation uses centered windows. Short sequences are rejected so
that no silent temporal padding changes the protocol.

If the Human3.6M archives are already arranged in the supplied D3DP format,
extract that repository and convert its camera-space data with:

```bash
python tools/prepare_h36m_from_d3dp.py \
  --d3dp-root /path/to/D3DP-main \
  --output-dir data \
  --keypoints cpn_ft_h36m_dbb \
  --frames 243 --stride 243
```

The converter imports D3DP's MIT-licensed dataset and camera utilities at run
time, converts every 3D sequence to root-relative camera coordinates, and
normalizes the CPN detections using the corresponding camera resolution.

Create a small synthetic dataset for a complete pipeline check:

```bash
python tools/make_synthetic_data.py --output-dir data/synthetic --frames 27
python train.py --config configs/smoke.yaml --dry-run
```

## Human3.6M training

Place prepared arrays at `data/h36m_train.npz` and `data/h36m_valid.npz`, then
run:

```bash
python train.py --config configs/h36m/hscd_b.yaml
```

The paper setting uses subjects S1, S5, S6, S7, and S8 for training and S9 and
S11 for evaluation, 243-frame inputs, CPN detections, AdamW for 100 epochs, a
batch size of 16, an initial learning rate of `5e-5`, and exponential decay of
0.99 per epoch.

## Evaluation

```bash
python evaluate.py \
  --config configs/h36m/hscd_b.yaml \
  --checkpoint runs/hscd_b_h36m/best.pt \
  --data data/h36m_valid.npz \
  --save-predictions predictions_h36m.npz
```

`evaluate.py` reports MPJPE and P-MPJPE. The default sampler performs 50
respaced reverse transitions; `--reverse-steps` can override it for the
accuracy/latency study.

For MPI-INF-3DHP, prepare the same NPZ keys with 81-frame windows and run
`configs/mpi3dhp/hscd.yaml`. That configuration follows the paper's 120-epoch,
batch-size-32, `5e-4` learning-rate, and ground-truth-2D setting. The evaluator
reports MPJPE, PCK@150 mm, and normalized AUC over 0--150 mm. Official benchmark
submission may still require the dataset's sequence/joint remapping; the
generic NPZ loader intentionally does not guess that mapping.

## Tests

```bash
pytest
```

The tests cover tensor shapes, MIE caching, scale-conditioned denoising,
generalized reverse transitions, metrics, and the NPZ dataset interface.

## Repository structure

```text
HSCD-main/
  configs/                 paper and smoke-test configurations
  hscd/
    models/                MIE, CCB, PRD, and the composed HSCD network
    config.py              validated YAML loading and model construction
    data.py                NPZ pose-sequence dataset
    diffusion.py           training corruption and respaced reverse process
    metrics.py             MPJPE and P-MPJPE
  tools/                   synthetic-data helper
  tests/                   unit and end-to-end smoke tests
  train.py                 training entry point
  evaluate.py              evaluation entry point
```
