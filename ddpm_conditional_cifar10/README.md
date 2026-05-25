# ddpm-conditional-cifar10

Standalone DDPM conditional image generation project for CIFAR-10, built as a clean sibling project under the current workspace without modifying the original Flow Matching code.

## Install

```bash
pip install -e .
```

## Train

```bash
export OPENAI_LOGDIR=/path/to/ddpm_cond_runs/train
python scripts/train.py \
  --data_dir /path/to/data \
  --dataset_name cifar10 \
  --cifar10_download true \
  --class_cond true \
  --num_classes 10 \
  --image_size 32 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --lr 1e-4 \
  --batch_size 64 \
  --microbatch 16 \
  --save_interval 5000 \
  --lr_anneal_steps 50000
```

You can also use an image folder where class names are inferred from directory names or filename prefixes such as `airplane_00001.png`:

```bash
python scripts/train.py \
  --data_dir /path/to/cifar_train \
  --dataset_name image_folder \
  --class_cond true \
  --num_classes 10 \
  --image_size 32
```

## Sample

```bash
export OPENAI_LOGDIR=/path/to/ddpm_cond_runs/sample
python scripts/sample.py \
  --model_path /path/to/train/model050000.pt \
  --class_cond true \
  --num_classes 10 \
  --image_size 32 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --batch_size 128 \
  --num_samples 1000 \
  --class_label -1
```

`--class_label -1` cycles through classes evenly. Set a non-negative label to sample a fixed class.
