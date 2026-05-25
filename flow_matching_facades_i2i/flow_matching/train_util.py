import copy

import blobfile as bf
import numpy as np
import torch as th
from torch.optim import AdamW

from . import dist_util, logger
from .nn import update_ema


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        flow,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.flow = flow
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.model_params = list(self.model.parameters())

        self._load_parameters()
        self.opt = AdamW(self.model_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(self.model_params) for _ in self.ema_rate]

    def _load_parameters(self):
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(self.resume_checkpoint, map_location=dist_util.dev())
            )
        dist_util.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        opt_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), f"opt{self.resume_step:06d}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.model_params)
        ema_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), f"ema_{rate}_{self.resume_step:06d}.pt"
        )
        if bf.exists(ema_checkpoint):
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return ema_params

    def run_loop(self):
        self.model.train()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            batch = batch.to(dist_util.dev())
            cond = self._move_cond_to_device(cond)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond=None):
        self.model.zero_grad(set_to_none=True)
        batch_size = batch.shape[0]
        loss_accum = 0.0
        for start in range(0, batch_size, self.microbatch):
            micro = batch[start : start + self.microbatch]
            micro_cond = self._slice_cond(cond, start, start + self.microbatch)
            losses = self.flow.training_losses(self.model, micro, model_kwargs=micro_cond)
            scale = micro.shape[0] / batch_size
            loss = losses["loss"].mean() * scale
            loss.backward()
            loss_accum += losses["loss"].mean().item() * scale
            logger.logkv_mean("loss", losses["loss"].mean().item())
            logger.logkv_mean("mse", losses["mse"].mean().item())

        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

        logger.logkv("step", self.step + self.resume_step)
        logger.logkv(
            "samples", (self.step + self.resume_step + 1) * self.global_batch
        )
        logger.logkv("loss_step", loss_accum)

    def _move_cond_to_device(self, cond):
        if not cond:
            return {}
        return {key: value.to(dist_util.dev()) for key, value in cond.items()}

    def _slice_cond(self, cond, start, end):
        if not cond:
            return {}
        return {key: value[start:end] for key, value in cond.items()}

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.model_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        logger.logkv("lr", lr)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._params_to_state_dict(params)
            if not rate:
                filename = f"model{self.step + self.resume_step:06d}.pt"
            else:
                filename = f"ema_{rate}_{self.step + self.resume_step:06d}.pt"
            logger.log(f"saving model {rate} to {filename}...")
            with bf.BlobFile(bf.join(logger.get_dir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.model_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        with bf.BlobFile(
            bf.join(logger.get_dir(), f"opt{self.step + self.resume_step:06d}.pt"), "wb"
        ) as f:
            th.save(self.opt.state_dict(), f)

    def _params_to_state_dict(self, params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            state_dict[name] = params[i]
        return state_dict


def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
