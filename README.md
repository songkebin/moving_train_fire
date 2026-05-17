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

The default configuration uses strict image/spreadsheet alignment. If an Excel sheet
has a different row count than its image directory, training stops with a detailed
`DataAlignmentError`.

## Predict

```bash
python -m move_train.predict --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

For one image:

```bash
python -m move_train.predict \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --image "data/0/0-1-1/0-1-1 (1).png" \
  --env 0 --speed 1 --hrr 1 --position 0.1
```

Predictions are written to `outputs/predictions/` as CSV heatmaps and PNG figures.
