import torch
import torch.nn as nn
from .interfaces import EncoderDecoderINR
from alpine.models.nonlin import Sine
from alpine.models.strainer import Strainer, get_linear_layer

class STRAINER_INR(Strainer, EncoderDecoderINR):
    def __init__(self, *args, decoder_hidden_features: int = None, **kwargs):
        """
        Alpine's Strainer with an optional separate decoder width. Alpine gives
        every layer hidden_features; with decoder_hidden_features the heads
        are rebuilt as encoder width -> decoder width -> ... -> out_features,
        with Alpine's own SIREN init and the same omegas. Left at None (or
        equal to hidden_features) the model is exactly Alpine's.
        """
        super().__init__(*args, **kwargs)
        width = kwargs["hidden_features"]
        if decoder_hidden_features and decoder_hidden_features != width:
            self.decoder = self._build_heads(width, decoder_hidden_features, kwargs["out_features"],
                                             kwargs["num_decoders"], kwargs.get("bias", True))

    def _build_heads(self, in_width: int, width: int, out_features: int, num_heads: int,
                     bias: bool) -> nn.ModuleList:
        n = self.num_decoder_layers
        omegas = self.omegas[len(self.omegas) - n:]
        heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []
            for j in range(n):
                last = j == n - 1
                layers.append(get_linear_layer(in_width if j == 0 else width,
                                               out_features if last else width,
                                               omega=omegas[j], bias=bias, is_last=last))
                if not last:
                    layers.append(Sine(omega=omegas[j]))
            heads.append(nn.ModuleList(layers))
        return heads

    def forward(self, coords, decoder_index: int = 0):
        """Shared encoder, then one decoder head. Alpine's own forward runs
        every head on every point, which jointly training N heads on N
        different cases (each with its own coordinates) does not want."""
        out = coords
        for layer in self.encoder:
            out = layer(out)
        for layer in self.decoder[decoder_index]:
            out = layer(out)
        return out

    def encoder_parameters(self) -> list:
        return list(self.encoder.parameters())

    def decoder_parameters(self) -> list:
        return list(self.decoder.parameters())

    def reset_decoder(self) -> None:
        # nn.Linear.reset_parameters() would apply PyTorch's default init, not
        # SIREN's, so copy in weights from fresh layers built by Alpine's own
        # initializer instead (same omegas and first/last-layer rules).
        with torch.no_grad():
            for decoder in self.decoder:
                linears = [m for m in decoder if isinstance(m, nn.Linear)]
                for j, layer in enumerate(linears):
                    fresh = get_linear_layer(
                        layer.in_features, layer.out_features,
                        omega=self.omegas[len(self.omegas) - len(linears) + j],
                        bias=layer.bias is not None,
                        is_last=(j == len(linears) - 1),
                    )
                    layer.weight.copy_(fresh.weight)
                    if layer.bias is not None:
                        layer.bias.copy_(fresh.bias)

    def encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()

    def decoder_state_dict(self) -> dict:
        return self.decoder.state_dict()

    def load_decoder_state_dict(self, state_dict: dict) -> None:
        self.decoder.load_state_dict(state_dict)

    def load_encoder_state_dict(self, state_dict: dict, freeze: bool = False) -> None:
        super().load_encoder_weights(state_dict)
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
