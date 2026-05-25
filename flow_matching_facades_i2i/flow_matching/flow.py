import torch as th

from .nn import mean_flat


class FlowMatching:
    def sample_time(self, batch_size, device):
        return th.rand(batch_size, device=device)

    def sample_path(self, data_batch, t, noise=None):
        if noise is None:
            noise = th.randn_like(data_batch)
        t_view = t.view(-1, *([1] * (data_batch.ndim - 1)))
        x_t = (1.0 - t_view) * noise + t_view * data_batch
        target_velocity = data_batch - noise
        return x_t, target_velocity

    def training_losses(self, model, data_batch, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        t = self.sample_time(data_batch.shape[0], data_batch.device)
        x_t, target_velocity = self.sample_path(data_batch, t, noise=noise)
        pred_velocity = model(x_t, t, **model_kwargs)
        mse = mean_flat((pred_velocity - target_velocity) ** 2)
        return {"loss": mse, "mse": mse}

    @th.no_grad()
    def sample_euler(
        self,
        model,
        shape,
        device,
        steps,
        model_kwargs=None,
        noise=None,
        clip_output=True,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            x = th.randn(shape, device=device)
        else:
            x = noise.to(device)

        dt = 1.0 / steps
        for step in range(steps):
            t = th.full((shape[0],), step / steps, device=device)
            x = x + dt * model(x, t, **model_kwargs)

        if clip_output:
            x = x.clamp(-1, 1)
        return x
