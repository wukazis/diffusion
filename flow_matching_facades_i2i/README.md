# flow-matching-facades-i2i

Standalone minimal Flow Matching project for paired image-to-image translation.

The project is designed for pix2pix-style datasets such as `facades`, where each file is a
single image composed of a left half and a right half. By default, the model learns
`left -> right`. Use `--pair_direction right_to_left` to reverse the translation
direction.

## Expected Dataset Layout

```text
facades/
  train/
    0001.jpg
    0002.jpg
  val/
    0001.jpg
  test/
    0001.jpg
```

Each file should contain a horizontally concatenated pair image.

The Berkeley pix2pix `facades` dataset uses this exact format.

Download:

```bash
mkdir -p /diffusion/datasets
cd /diffusion/datasets
wget https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
tar -xzf facades.tar.gz
```

## Install

```bash
pip install -e .
```

## Train

```bash
export OPENAI_LOGDIR=/path/to/fm_runs/facades_train
python scripts/train.py \
  --data_dir /path/to/facades \
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
export OPENAI_LOGDIR=/path/to/fm_runs/facades_sample
python scripts/sample.py \
  --data_dir /path/to/facades \
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

## Evaluate

For paired translation tasks, `PSNR` and `SSIM` are usually more informative than FID because
they compare each prediction to its matched ground-truth target.

Preferred usage: evaluate the `samples_*.npz` file saved by `scripts/sample.py`.

```bash
python scripts/evaluate.py \
  --samples_npz /path/to/facades_sample/samples_16x256x256x3.npz
```

If you only copied the triplet PNGs, the script can also split `[condition | target | prediction]`
panels automatically:

```bash
python scripts/evaluate.py \
  --triplet_dir /path/to/facades_sample/triplet_panels
```

Optional perceptual metric:

```bash
pip install lpips
python scripts/evaluate.py \
  --samples_npz /path/to/facades_sample/samples_16x256x256x3.npz \
  --compute_lpips true
```

The script writes:

- `metrics_summary.json`
- `metrics_per_sample.csv`
