"""
Model interfaces.

Training code in this package depends only on this Protocol, not on any
specific INR library. This keeps the option open to plug in Alpine's
`Strainer`/`Siren` classes (recommended starting point, see project README)
or a hand-rolled SIREN without touching the training loop.
"""
from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class EncoderDecoderINR(Protocol):
    """
    Minimal contract a shared-encoder / per-image-decoder INR must satisfy.

    coords: (N, 3) tensor of coordinates in [-1, 1]
    returns: (N, out_features) tensor -- intensity (out_features=1) or SDF (out_features=1)
    """

    def forward(self, coords: torch.Tensor) -> torch.Tensor: ...

    def encoder_parameters(self) -> "list[torch.nn.Parameter]": ...

    def decoder_parameters(self) -> "list[torch.nn.Parameter]": ...

    def reset_decoder(self) -> None:
        """Reinitialize only the decoder. Encoder weights must be untouched.

        This is the mechanism behind 'decoder replaced per image, encoder
        kept across images' from Phase I of the architecture.
        """
        ...

    def encoder_state_dict(self) -> dict: ...

    def decoder_state_dict(self) -> dict: ...

    def load_encoder_state_dict(self, state_dict: dict, freeze: bool = False) -> None: ...


def build_model(model_cfg: dict, out_features: int) -> EncoderDecoderINR:
    """
    Factory for an EncoderDecoderINR from the `model:` block of the training
    config. This is the one place a concrete architecture needs to be wired
    in -- e.g. Alpine's `alpine.models.Strainer`.

    Raises NotImplementedError until a concrete backend is plugged in, so
    the rest of the pipeline (data, losses, metrics, training loop, the
    translation test) can be built and unit-tested independently of that
    decision.
    """
    raise NotImplementedError(
        "Wire a concrete EncoderDecoderINR here, e.g. by wrapping "
        "alpine.models.Strainer(encoder_layers=model_cfg['encoder_layers'], "
        "decoder_layers=model_cfg['decoder_layers'], hidden_dim=model_cfg['hidden_dim'], "
        f"out_features={out_features}) so it satisfies EncoderDecoderINR above."
    )
