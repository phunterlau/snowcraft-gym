"""R1j matched deterministic displacement/direction decoder corrections."""

import torch
from torch import Tensor
from torch.nn import functional as F

from ..ppo import living_unit_mask
from .geometry_probe import GeometryProbe
from .model import select_action_target

ARMS = ("absolute", "displacement", "direction", "both")


def clip_along_ray(origin: Tensor, target: Tensor) -> Tensor:
    """Clip to normalized arena box along origin-to-target ray, not per axis."""
    delta = target - origin
    active = delta.abs() > 1e-12
    denominator = torch.where(active, delta, torch.ones_like(delta))
    boundary = torch.where(delta > 0, torch.ones_like(delta), -torch.ones_like(delta))
    ratio = torch.where(active, (boundary-origin)/denominator, torch.ones_like(delta))
    fraction = ratio.amin(-1, keepdim=True).clamp(0, 1)
    # Preserve in-bounds targets bit-for-bit, including the source at initialization.
    return torch.where((target.abs() <= 1).all(-1, keepdim=True), target,
                       (origin + fraction * delta).clamp(-1, 1))


def direction_target(origin: Tensor, inherited: Tensor, residual: Tensor) -> Tensor:
    scale = origin.new_tensor([50., 40.])
    ray = (inherited-origin) * scale
    length = ray.norm(dim=-1, keepdim=True)
    unit = torch.where(length > 1e-6, ray / length.clamp_min(1e-6), torch.zeros_like(ray))
    # normalize(unit) is used in the subtraction as well, so zero corrections
    # cancel exactly despite float32 unit-length rounding.
    reference = F.normalize(unit, dim=-1, eps=1e-6)
    corrected = F.normalize(unit + residual, dim=-1, eps=1e-6)
    radius = torch.where(length > 1e-6, length, torch.ones_like(length))
    target = inherited + (corrected-reference) * radius / scale
    return clip_along_ray(origin, target)


class DecoderProbe(GeometryProbe):
    def __init__(self, source, *, arm: str):
        if arm not in ARMS:
            raise ValueError("unknown decoder arm")
        super().__init__(source, relative=False)
        self.arm = arm

    def forward(self, observation):
        if self.arm == "absolute":
            return super().forward(observation)
        with torch.no_grad():
            inherited = self.source(observation)
        features = self.features(observation)
        live = living_unit_mask(observation)[..., None]
        move, shot = self.move(features) * live, self.shot(features) * live
        base = inherited["target_raw_by_action"]
        move_target = (inherited["target_by_action"][..., 1, :] +
                       10 * torch.tanh(move) / move.new_tensor([50., 40.])).clamp(-1, 1)
        if self.arm not in {"displacement", "both"}:
            move_target = torch.tanh(base[..., 1, :] + move)
        throw_target = direction_target(observation["allies"][..., 2:4],
                                         inherited["target_by_action"][..., 2, :], shot[..., :2])
        if self.arm not in {"direction", "both"}:
            throw_target = torch.tanh(base[..., 2, :] + shot[..., :2])
        targets = torch.stack([inherited["target_by_action"][..., 0, :], move_target,
                               throw_target, inherited["target_by_action"][..., 3, :]], -2)
        selected = inherited["action_logits"].argmax(-1)
        power_raw = inherited["power_raw"] + shot[..., 2]
        result = {**inherited, "target_by_action": targets, "supervised_target_by_action": targets,
                  "target": select_action_target(targets, selected), "power_raw": power_raw,
                  "power": torch.sigmoid(power_raw)}
        # These decoders have no inverse tanh action-mean contract. Do not expose
        # stale inherited means to a stochastic collector.
        for key in ("target_raw", "target_raw_by_action", "base_target_raw_by_action"):
            result.pop(key, None)
        return result
