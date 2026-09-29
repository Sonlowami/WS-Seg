from .interfaces import EncoderDecoderINR
from alpine.models.strainer import Strainer

class STRAINER_INR(Strainer, EncoderDecoderINR):
    def forward(self, coords):
        return super().forward(coords)["output"].squeeze(1)

    def encoder_parameters(self) -> list:
        return list(self.encoder.parameters())

    def decoder_parameters(self) -> list:
        return list(self.decoder.parameters())

    def reset_decoder(self) -> None:
        for module in self.decoder.modules():
            reset_parameters = getattr(module, "reset_parameters", None)
            if reset_parameters is not None:
                reset_parameters()

    def encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()

    def decoder_state_dict(self) -> dict:
        return self.decoder.state_dict()

    def load_encoder_state_dict(self, state_dict: dict, freeze: bool = False) -> None:
        super().load_encoder_weights(state_dict)
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
