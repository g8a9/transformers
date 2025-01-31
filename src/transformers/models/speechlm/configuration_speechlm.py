from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class SpeechLMConfig(PretrainedConfig):

    model_type = "speech-encoder-decoder"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because not both `encoder` and"
                f" `decoder` sub-configurations are passed, but only {kwargs}"
            )

        # encoder_config = AutoConfig.from_pretrained(kwargs.pop("encoder"))
        # encoder_model_type = encoder_config.pop("model_type")
        # decoder_config = AutoConfig.from_pretrained(kwargs.pop("decoder"))
        # decoder_model_type = decoder_config.pop("model_type")

        # self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        # self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.encoder = AutoConfig.from_pretrained(kwargs.pop("encoder"))
        self.decoder = AutoConfig.from_pretrained(kwargs.pop("decoder"))
        self.is_encoder_decoder = True

        self.decoder_start_token_id = self.decoder.bos_token_id
        self.pad_token_id = self.decoder.eos_token_id

    # @classmethod
    # def from_encoder_decoder_configs(
    #     cls,
    #     encoder_config: PretrainedConfig,
    #     decoder_config: PretrainedConfig,
    #     **kwargs,
    # ) -> PretrainedConfig:
    #     r"""
    #     Instantiate a [`SpeechEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
    #     configuration and decoder model configuration.

    #     Returns:
    #         [`SpeechEncoderDecoderConfig`]: An instance of a configuration object
    #     """
    #     logger.info(
    #         "Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config"
    #     )
    #     decoder_config.is_decoder = True
    #     decoder_config.add_cross_attention = True

    #     return cls(
    #         encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs
    #     )


__all__ = ["SpeechLMConfig"]
