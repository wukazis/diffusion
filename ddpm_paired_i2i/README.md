# ddpm-paired-i2i

Standalone DDPM project for paired image-to-image translation built from the
`improved_diffusion` baseline.

The project is designed for pix2pix-style datasets where each file contains a
left half and a right half. The same code can be used for `maps`, `facades`,
`edges2shoes`, and similar paired datasets.

## Expected Dataset Layout

```text
dataset_name/
  train/
    0001.jpg
    0002.jpg
  val/
    0001.jpg
  test/
    0001.jpg
```

Each image file should contain a horizontally concatenated pair image.

## Install

```bash
pip install -e .
```

## Train

Example for `maps` (`satellite -> map`):

```bash
export OPENAI_LOGDIR=/path/to/ddpm_runs/maps_train
python scripts/train.py \
  --data_dir /path/to/maps \
  --train_split train \
  --pair_direction left_to_right \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --lr 1e-4 \
  --batch_size 4 \
  --microbatch 1 \
  --save_interval 5000 \
  --lr_anneal_steps 50000
```

Example for `facades` (`label -> photo`):

```bash
export OPENAI_LOGDIR=/path/to/ddpm_runs/facades_train
python scripts/train.py \
  --data_dir /path/to/facades \
  --train_split train \
  --pair_direction right_to_left \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --lr 1e-4 \
  --batch_size 4 \
  --microbatch 1 \
  --save_interval 5000 \
  --lr_anneal_steps 50000
```

## Sample

Sampling uses real condition images from a split such as `val` and generates a
translated output for each one.

```bash
export OPENAI_LOGDIR=/path/to/ddpm_runs/maps_sample
python scripts/sample.py \
  --data_dir /path/to/maps \
  --sample_split val \
  --pair_direction left_to_right \
  --model_path /path/to/train/model050000.pt \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --batch_size 4 \
  --num_samples 16
```

For DDIM sampling:

```bash
python scripts/sample.py \
  --data_dir /path/to/maps \
  --sample_split val \
  --pair_direction left_to_right \
  --model_path /path/to/train/model050000.pt \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --use_ddim true \
  --timestep_respacing ddim100 \
  --batch_size 4 \
  --num_samples 16
```

Outputs are saved to the logging directory:

- `generated_pngs/`: generated images only
- `triplet_panels/`: `[condition | target | prediction]` panels
- `sample_grid.png`: grid of generated outputs
- `triplet_grid.png`: grid of triplet panels
- `samples_*.npz`: NumPy archive with `cond`, `target`, and `pred`

## Evaluate

For paired translation tasks, use `PSNR`, `SSIM`, and optionally `LPIPS`.

```bash
python scripts/evaluate.py \
  --samples_npz /path/to/sample_run/samples_16x256x256x3.npz
```

Or evaluate directly from copied triplet panels:

```bash
python scripts/evaluate.py \
  --triplet_dir /path/to/sample_run/triplet_panels
```
