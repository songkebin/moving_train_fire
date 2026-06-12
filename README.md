# Move Train Fire Transformer Prototype

This repository contains experiment data and a PyTorch prototype for predicting a `5x9`
train-roof temperature field from flame images and physical condition tokens.

## Setup

```bash
conda env create -f environment.yml
conda activate move-train
```

## Train

```bash
python -m move_train.train --config configs/default.yaml
```

The default configuration reads `data/manifests/samples.csv`, trains on a random
sample-level split of all constant-speed data, and tests on all deceleration data:

- `data/splits/constant_random_train.csv`
- `data/splits/constant_random_val.csv`
- `data/splits/deceleration_test.csv`

Source-domain testing and transfer evaluation splits are prepared under
`data/splits/`.

## Predict

```bash
python -m move_train.predict --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

For one image:

```bash
python -m move_train.predict \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --image "data/raw/constant_tunnel/0-1-1/0-1-1 (1).png" \
  --env 0 --speed 1 --hrr 0 \
  --speed-value 1.0 --hrr-value 1.0 \
  --position 0.1
```

Predictions are written to `outputs/predictions/` as CSV heatmaps and PNG figures.
