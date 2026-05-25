# flow-matching-cifar10-conditional-v2

Enhanced standalone conditional Flow Matching project for CIFAR-10 style `32x32` generation.

## What Changed

- Stronger class conditioning with scale-shift modulation inside residual blocks.
- Classifier-free guidance style training via class dropout and a learned null class token.
- Added Heun sampling alongside Euler sampling.
- Fixed `image_folder` workflows so actual class names are saved in training metadata and reused at sampling time.
- Sampling now supports both `--class_label` and `--class_name`.

## Recommended Use

Use this version when your training set is a flat folder of images named like:

```text
bird_00001.png
car_00002.png
plane_00003.png
```

For `image_folder`, class ids follow the sorted prefix order from filenames. For a CIFAR-10 style set with prefixes

```text
bird car cat deer dog frog horse plane ship truck
```

the mapping is:

- `0` bird
- `1` car
- `2` cat
- `3` deer
- `4` dog
- `5` frog
- `6` horse
- `7` plane
- `8` ship
- `9` truck

You can avoid remembering this by sampling with `--class_name`.

## Install

```bash
pip install -e .
```

## Train

Linux server example for a flat image folder dataset:

```bash
cd /diffusion/flow_matching_cifar10_conditional_v2
pip install -e .

export PYTHONPATH=$(pwd)
export OPENAI_LOGDIR=/root/autodl-tmp/exp/fm_cifar10_cond_v2_train

python scripts/train.py \
  --data_dir /diffusion/datasets/cifar_train \
  --dataset_name image_folder \
  --image_size 32 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --use_attention true \
  --use_scale_shift_norm true \
  --class_cond true \
  --num_classes 10 \
  --class_dropout_prob 0.1 \
  --lr 1e-4 \
  --batch_size 128 \
  --microbatch 32 \
  --save_interval 5000 \
  --lr_anneal_steps 50000
```

If `50k` is still weak, continue to `100k`.

## Sample

Recommended baseline sampling:

```bash
cd /diffusion/flow_matching_cifar10_conditional_v2
export PYTHONPATH=$(pwd)
export OPENAI_LOGDIR=/root/autodl-tmp/exp/fm_cifar10_cond_v2_fid10k

python scripts/sample.py \
  --model_path /root/autodl-tmp/exp/fm_cifar10_cond_v2_train/model050000.pt \
  --image_size 32 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --use_attention true \
  --use_scale_shift_norm true \
  --class_cond true \
  --num_classes 10 \
  --sampling_steps 250 \
  --sampling_method heun \
  --guidance_scale 1.5 \
  --batch_size 128 \
  --num_samples 10000 \
  --class_label -1 \
  --save_png_samples 10000 \
  --save_grid false
```

Sample a fixed class by name:

```bash
python scripts/sample.py \
  --model_path /root/autodl-tmp/exp/fm_cifar10_cond_v2_train/model050000.pt \
  --image_size 32 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --class_cond true \
  --num_classes 10 \
  --sampling_steps 250 \
  --sampling_method heun \
  --guidance_scale 1.5 \
  --batch_size 64 \
  --num_samples 64 \
  --class_name plane
```

If guidance starts to oversaturate or collapse samples, try:

- `--guidance_scale 1.2`
- `--guidance_scale 1.0`

If samples are still too rough, try:

- `--sampling_steps 500`

## Notes

- Prefer `model050000.pt` first. On short FM runs, EMA is not always better.
- For `image_folder`, this version saves the actual class order into `train_run_metadata.json`.
- Sampling metadata also records the resolved class names, sampled labels, runtime, and throughput.
