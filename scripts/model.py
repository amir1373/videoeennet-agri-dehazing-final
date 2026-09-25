"""VideoEENet: frequency/spatial encoder with ConvLSTM temporal memory."""
from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class FrequencyProcessingModule(nn.Module):
    """Learnable real/imaginary FFT processing used by the EENet encoder."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real, self.imag, self.fuse = nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Float32 FFT is reliable under CUDA AMP, including non-power-of-two inputs.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
            restored = torch.fft.ifft2(torch.complex(self.real(spectrum.real), self.imag(spectrum.imag)), dim=(-2, -1), norm="ortho").real
            return self.fuse(restored)

class DualDomainBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.frequency, self.spatial, self.fuse = FrequencyProcessingModule(channels), nn.Sequential(ResidualBlock(channels), ResidualBlock(channels)), nn.Conv2d(channels * 2, channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.frequency(x), self.spatial(x)], dim=1)) + x

class EENetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.stage1, self.down1 = DualDomainBlock(base_channels), nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.stage2, self.down2 = DualDomainBlock(base_channels * 2), nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.stage3, self.out_channels = DualDomainBlock(base_channels * 4), base_channels * 4
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage3(self.down2(self.stage2(self.down1(self.stage1(self.stem(x))))))

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim, self.gates = hidden_dim, nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)
    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cell, self.hidden_dim = ConvLSTMCell(input_dim, hidden_dim), hidden_dim
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5: raise ValueError(f"Expected [B, T, C, H, W], got {tuple(sequence.shape)}")
        batch, _, _, height, width = sequence.shape
        h = sequence.new_zeros(batch, self.hidden_dim, height, width); c = torch.zeros_like(h)
        for frame in sequence.unbind(dim=1): h, c = self.cell(frame, (h, c))
        return h

class SpatialTemporalTransformer(nn.Module):
    """Temporal self-attention independently at each spatial patch location.

    Pooling creates a patch grid, never a single global-frame token. Each patch
    attends only over its T observations, preserving the spatial layout needed
    for fair ConvLSTM-versus-transformer ablations.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, pool_size: int = 8, max_seq_len: int = 10) -> None:
        super().__init__()
        if hidden_dim % num_heads: raise ValueError("hidden_dim must be divisible by num_heads")
        self.pool_size, self.max_seq_len = pool_size, max_seq_len
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Parameter(torch.zeros(1, max_seq_len, 1, hidden_dim))
        self.patch_embedding = nn.Parameter(torch.zeros(1, 1, pool_size * pool_size, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder, self.norm = nn.TransformerEncoder(layer, num_layers=2), nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.time_embedding, std=0.02); nn.init.normal_(self.patch_embedding, std=0.02)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = features.shape
        if steps > self.max_seq_len: raise ValueError(f"Sequence length {steps} exceeds max_seq_len={self.max_seq_len}")
        pooled = F.adaptive_avg_pool2d(features.reshape(batch * steps, channels, height, width), (self.pool_size, self.pool_size))
        tokens = self.proj(pooled.reshape(batch, steps, channels, -1).permute(0, 1, 3, 2))
        tokens = tokens + self.time_embedding[:, :steps] + self.patch_embedding
        tokens = self.encoder(tokens.permute(0, 2, 1, 3).reshape(batch * self.pool_size * self.pool_size, steps, -1))
        current = self.norm(tokens[:, -1]).reshape(batch, self.pool_size, self.pool_size, -1).permute(0, 3, 1, 2)
        return F.interpolate(current, size=(height, width), mode="bilinear", align_corners=False)

TEMPORAL_MODES = ("convlstm", "spatial_transformer", "hybrid", "single_frame")

class VideoEENet(nn.Module):
    """Predict the clean final frame from a hazy sequence [B, T, 3, H, W].

    Only the modules the chosen temporal_mode uses are constructed, so every parameter of a
    model is trained. "single_frame" is the no-temporal ablation: the identical network (same
    encoder, ConvLSTM and decoder, same parameter count) is given only the final frame, so the
    ConvLSTM runs for one step from a zero state and carries no temporal information.
    """
    def __init__(self, base_channels: int = 32, hidden_dim: int = 64, temporal_mode: str = "convlstm", attention_heads: int = 4, attention_pool_size: int = 8, max_seq_len: int = 10) -> None:
        super().__init__()
        if temporal_mode not in TEMPORAL_MODES: raise ValueError(f"temporal_mode must be one of {TEMPORAL_MODES}")
        self.temporal_mode = temporal_mode
        self.encoder = EENetEncoder(base_channels=base_channels)
        if temporal_mode in {"convlstm", "hybrid", "single_frame"}:
            self.temporal = ConvLSTM(base_channels * 4, hidden_dim)
        if temporal_mode in {"spatial_transformer", "hybrid"}:
            self.attention = SpatialTemporalTransformer(base_channels * 4, hidden_dim, attention_heads, attention_pool_size, max_seq_len)
        if temporal_mode == "hybrid":
            self.hybrid_fuse = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        self.decoder = nn.Sequential(nn.Conv2d(hidden_dim + base_channels * 4, base_channels * 4, 3, padding=1), nn.GELU(), nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1), nn.GELU(), nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1), nn.GELU(), nn.Conv2d(base_channels, 3, 3, padding=1))
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3: raise ValueError(f"Expected [B, T, 3, H, W], got {tuple(frames.shape)}")
        if self.temporal_mode == "single_frame":
            frames = frames[:, -1:]
        batch, steps, _, height, width = frames.shape
        if steps < 2 and self.temporal_mode != "single_frame": raise ValueError("VideoEENet requires at least two temporal frames.")
        pad_h, pad_w = (-height) % 4, (-width) % 4
        padded = F.pad(frames.reshape(-1, 3, height, width), (0, pad_w, 0, pad_h), mode="replicate")
        features = self.encoder(padded).reshape(batch, steps, -1, (height + pad_h) // 4, (width + pad_w) // 4)
        if self.temporal_mode in {"convlstm", "single_frame"}:
            memory = self.temporal(features)
        elif self.temporal_mode == "spatial_transformer":
            memory = self.attention(features)
        else:
            memory = self.hybrid_fuse(torch.cat([self.temporal(features), self.attention(features)], dim=1))
        residual = self.decoder(torch.cat([memory, features[:, -1]], dim=1))
        current = padded.reshape(batch, steps, 3, height + pad_h, width + pad_w)[:, -1]
        return torch.sigmoid(residual + current)[:, :, :height, :width]

def load_model_state(model: VideoEENet, state: dict) -> list[str]:
    """Load a checkpoint, including ones saved before unused modules were removed.

    Checkpoints from earlier versions also hold the never-used attention/hybrid_fuse weights of
    a convlstm model; those keys are dropped (and returned) and everything else must match.
    """
    own = set(model.state_dict())
    dropped = [key for key in state if key not in own and key.split(".")[0] in {"attention", "hybrid_fuse"}]
    model.load_state_dict({key: value for key, value in state.items() if key not in dropped}, strict=True)
    return dropped

def shape_test() -> dict[str, tuple[int, ...]]:
    frames = torch.rand(1, 10, 3, 64, 64); results = {"frames": tuple(frames.shape)}
    for mode in TEMPORAL_MODES:
        with torch.no_grad(): results[mode] = tuple(VideoEENet(base_channels=8, hidden_dim=16, temporal_mode=mode).eval()(frames).shape)
    return results
