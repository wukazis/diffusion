# flow-matching-project

Standalone minimal Flow Matching project for CIFAR-10 image generation.

## Install

```bash
pip install -e .
```

## Train

```bash
export OPENAI_LOGDIR=/path/to/fm_runs/train
python scripts/train.py \
  --data_dir /path/to/cifar_train \
  --image_size 32 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --lr 1e-4 \
  --batch_size 64 \
  --microbatch 16 \
  --save_interval 5000 \
  --lr_anneal_steps 50000
```

## Sample

```bash
export OPENAI_LOGDIR=/path/to/fm_runs/sample
python scripts/sample.py \
  --model_path /path/to/train/ema_0.9999_050000.pt \
  --image_size 32 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --sampling_steps 100 \
  --batch_size 128 \
  --num_samples 1000
```
