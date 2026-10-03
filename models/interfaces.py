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

    def forward(self, coords: torch.Tensor, decoder_index: int = 0) -> torch.Tensor:
        """Shared encoder, then decoder head `decoder_index` (joint training
        keeps one head per training case; everything else uses head 0)."""
        ...

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


def build_model(model_cfg: dict, out_features: int, num_decoders: int = 1) -> EncoderDecoderINR:
    """
    Build an EncoderDecoderINR from the `model:` block of the training config.
    This is the one place a concrete architecture is wired in; it wraps
    Alpine's `Strainer` (see models/strainer_inr.py).

    model_cfg keys:
        encoder_layers  shared layers kept across images
        decoder_layers  per-image layers, re-initialized for each case
        hidden_dim      width of every hidden layer
        omega           optional SIREN frequency, default 30.0
    Inputs are always 3-D coordinates; out_features is the target width
    (image channels for Encoder I, label groups for Encoder II).
    num_decoders: one head per training case for STRAINER-style joint
    training (training/train_loop.py: train_encoder_jointly); 1 otherwise.
    """
    # Imported here: strainer_inr imports this module for the Protocol.
    from models.strainer_inr import STRAINER_INR

    encoder_layers = model_cfg["encoder_layers"]
    decoder_layers = model_cfg["decoder_layers"]
    if encoder_layers < 1 or decoder_layers < 1:
        raise ValueError(
            f"Need at least one encoder and one decoder layer, got "
            f"encoder_layers={encoder_layers}, decoder_layers={decoder_layers}"
        )
    return STRAINER_INR(
        in_features=3,
        hidden_features=model_cfg["hidden_dim"],
        hidden_layers=encoder_layers + decoder_layers,
        out_features=out_features,
        num_shared_layers=encoder_layers,
        num_decoders=num_decoders,
        omegas=[float(model_cfg.get("omega", 30.0))],
    )


def print_model_summary(model: EncoderDecoderINR, n_points: int = 4096,
                        title: str = "STRAINER model") -> dict:
    """
    Print a torchinfo layer table (from a forward pass on n_points random
    coordinates) plus parameter counts split by encoder/decoder and by
    trainable/frozen, so a frozen encoder is visible at a glance.
    The model stays on its current device. Returns the counts.
    """
    from torchinfo import summary

    def count(params, trainable=None):
        return sum(p.numel() for p in params
                   if trainable is None or p.requires_grad == trainable)

    device = next(model.parameters()).device
    stats = summary(model, input_size=(n_points, 3), device=device, depth=3, verbose=0,
                    col_names=("input_size", "output_size", "num_params", "trainable"))
    params = list(model.parameters())
    counts = {
        "total": count(params),
        "encoder": count(model.encoder_parameters()),
        "decoder": count(model.decoder_parameters()),
        "trainable": count(params, trainable=True),
        "frozen": count(params, trainable=False),
    }
    print(f"\n{title}\n{stats}")
    print("Parameters: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()) + "\n")
    return counts
