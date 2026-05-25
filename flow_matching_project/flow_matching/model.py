import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import SiLU, linear, normalization, timestep_embedding, zero_module


class FMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout):
        super().__init__()
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(SiLU(), linear(emb_channels, out_channels))
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip_connection(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(x).chunk(3, dim=1)
        scale = c ** -0.5
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight, dim=-1)
        h_out = th.einsum("bts,bcs->bct", weight, v)
        h_out = self.proj_out(h_out).reshape(b, c, h, w)
        return x_in + h_out


class FMUNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        dropout=0.0,
        channel_mult=(1, 2, 2, 2),
        attention_levels=(1, 2),
    ):
        super().__init__()
        self.model_channels = model_channels
        self.in_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.encoder_attn = nn.ModuleList()
        self.skip_channels = []
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            level_ch = model_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(FMResBlock(ch, level_ch, time_embed_dim, dropout))
                ch = level_ch
            self.encoder_blocks.append(blocks)
            self.encoder_attn.append(
                AttentionBlock(ch) if level in attention_levels else nn.Identity()
            )
            self.skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))

        self.middle_blocks = nn.ModuleList(
            [FMResBlock(ch, ch, time_embed_dim, dropout), FMResBlock(ch, ch, time_embed_dim, dropout)]
        )
        self.middle_attn = AttentionBlock(ch)

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoder_attn = nn.ModuleList()
        for level in reversed(range(len(channel_mult))):
            skip_ch = self.skip_channels[level]
            blocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                block_in = ch + skip_ch if block_idx == 0 else skip_ch
                blocks.append(FMResBlock(block_in, skip_ch, time_embed_dim, dropout))
                ch = skip_ch
            self.decoder_blocks.append(blocks)
            self.decoder_attn.append(
                AttentionBlock(ch) if level in attention_levels else nn.Identity()
            )
            if level != 0:
                self.upsamples.append(Upsample(ch))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, t):
        if t.ndim == 0:
            t = t[None]
        emb = self.time_embed(timestep_embedding(t * 1000.0, self.model_channels))

        h = self.in_conv(x)
        skips = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                h = block(h, emb)
            h = self.encoder_attn[level](h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        for block in self.middle_blocks:
            h = block(h, emb)
        h = self.middle_attn(h)

        for level, blocks in enumerate(self.decoder_blocks):
            h = th.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, emb)
            h = self.decoder_attn[level](h)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        return self.out(h)
