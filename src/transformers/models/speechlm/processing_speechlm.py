# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Speech processor class for Wav2Vec2-BERT
"""

import warnings
from typing import List, Optional, Union

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from ..llama.tokenization_llama import LlamaTokenizer
from ... import AutoTokenizer
from tokenizers import AddedToken
import torch

import pdb


class SpeechLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class SpeechLMProcessor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2-BERT processor which wraps a Wav2Vec2-BERT feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`SeamlessM4TFeatureExtractor`):
            An instance of [`SeamlessM4TFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    lang2token = {
        "ita": "<|ita|>",
        "eng": "<|eng|>",
    }
    task2token = {
        "transcribe": "<|transcribe|>",
        "translate": "<|translate|>",
    }

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls, audio_model_name_or_path, text_model_name_or_path, **kwargs
    ):
        # try:
        #     return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # except OSError:
        #     warnings.warn(
        #         f"Loading a tokenizer inside {cls.__name__} from a config that does not"
        #         " include a `tokenizer_class` attribute is deprecated and will be "
        #         "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
        #         " attribute to either your `config.json` or `tokenizer_config.json` "
        #         "file to suppress this warning: ",
        #         FutureWarning,
        #     )

        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            audio_model_name_or_path, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path, **kwargs)

        additional_tokens = [
            *list(cls.task2token.values()),
            *list(cls.lang2token.values()),
        ]

        tokenizer.add_tokens(additional_tokens, special_tokens=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def _add_lang_task_tokens(self, text_inputs, lang, task):
        text_inputs["input_ids"] = [
            [t[0]]  # bos_token
            + [
                self.tokenizer.convert_tokens_to_ids(self.lang2token[lang]),
                self.tokenizer.convert_tokens_to_ids(self.task2token[task]),
            ]
            + t[1:]
            for t in text_inputs["input_ids"]
        ]
        text_inputs["attention_mask"] = [
            [1, 1] + am for am in text_inputs["attention_mask"]
        ]
        return text_inputs

    def __call__(
        self,
        audio: AudioInput,
        task: str,
        lang: str,
        text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = None,
        **kwargs,
    ):
        """
        TODO: update docstring
        """

        if audio is None and text is None:
            raise ValueError(
                "You need to specify either an `audio` or `text` input to process."
            )
        output_kwargs = self._merge_kwargs(
            SpeechLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # if audio is not None:
        audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        if text is not None:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        else:
            text_inputs = {
                "input_ids": [[self.tokenizer.bos_token_id]],
                "attention_mask": [[1]],
            }

        text_inputs = self._add_lang_task_tokens(text_inputs, lang, task)

        merged_inputs = {
            **{f"audio_{k}": v for k, v in audio_inputs.items()},
            **{f"text_{k}": v for k, v in text_inputs.items()},
        }

        return merged_inputs

    #     def pad(self, input_features=None, labels=None, **kwargs):
    #         """
    #         If `input_features` is not `None`, this method forwards the `input_features` and `kwargs` arguments to SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.pad`] to pad the input features.
    #         If `labels` is not `None`, this method forwards the `labels` and `kwargs` arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.pad`] to pad the label(s).
    #         Please refer to the doctsring of the above two methods for more information.
    #         """
    #         if input_features is None and labels is None:
    #             raise ValueError(
    #                 "You need to specify either an `input_features` or `labels` input to pad."
    #             )

    #         if input_features is not None:
    #             input_features = self.feature_extractor.pad(input_features, **kwargs)
    #         if labels is not None:
    #             labels = self.tokenizer.pad(labels, **kwargs)

    #         if labels is None:
    #             return input_features
    #         elif input_features is None:
    #             return labels
    #         else:
    #             input_features["labels"] = labels["input_ids"]
    #             return input_features

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


# __all__ = ["Wav2Vec2BertProcessor"]
