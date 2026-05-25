# flow-matching-maps-i2i

Standalone minimal Flow Matching project for paired image-to-image translation.

The project is designed for pix2pix-style datasets such as `maps`, where each file is a
single image composed of a left half and a right half. By default, the model learns
`left -> right`. Use `--pair_direction right_to_left` to reverse the translation
direction.

## Expected Dataset Layout

```text
maps/
  train/
    0001.jpg
    0002.jpg
  val/
    0001.jpg
  test/
    0001.jpg
```

Each file should contain a horizontally concatenated pair image.

## Install

```bash
pip install -e .
```

## Train

```bash
export OPENAI_LOGDIR=/path/to/fm_runs/maps_train
python scripts/train.py \
  --data_dir /path/to/maps \
  --train_split train \
  --pair_direction left_to_right \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --lr 1e-4 \
  --batch_size 4 \
  --microbatch 1 \
  --save_interval 2000 \
  --lr_anneal_steps 20000
```

## Sample

Sampling uses real condition images from a split such as `val` and generates translated
outputs for them.

```bash
export OPENAI_LOGDIR=/path/to/fm_runs/maps_sample
python scripts/sample.py \
  --data_dir /path/to/maps \
  --sample_split val \
  --pair_direction left_to_right \
  --model_path /path/to/train/model020000.pt \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --sampling_steps 100 \
  --batch_size 4 \
  --num_samples 16
```

Outputs are saved to the logging directory:

- `generated_pngs/`: generated images only
- `triplet_panels/`: `[condition | target | prediction]` panels
- `sample_grid.png`: grid of generated outputs
- `triplet_grid.png`: grid of triplet panels
- `samples_*.npz`: NumPy archive with `cond`, `target`, and `pred`
