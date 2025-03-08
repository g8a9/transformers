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
from typing import List, Optional, Union, Dict

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from ..llama.tokenization_llama import LlamaTokenizer
from ... import AutoTokenizer
from tokenizers import AddedToken
import torch
import itertools

import pdb
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        "bg": "<|bg|>",  # Bulgarian
        "hr": "<|hr|>",  # Croatian
        "cs": "<|cs|>",  # Czech
        "da": "<|da|>",  # Danish
        "nl": "<|nl|>",  # Dutch
        "en": "<|en|>",  # English
        "et": "<|et|>",  # Estonian
        "fi": "<|fi|>",  # Finnish
        "fr": "<|fr|>",  # French
        "de": "<|de|>",  # German
        "el": "<|el|>",  # Greek
        "hu": "<|hu|>",  # Hungarian
        "ga": "<|ga|>",  # Irish
        "it": "<|it|>",  # Italian
        "lv": "<|lv|>",  # Latvian
        "lt": "<|lt|>",  # Lithuanian
        "mt": "<|mt|>",  # Maltese
        "pl": "<|pl|>",  # Polish
        "pt": "<|pt|>",  # Portuguese
        "ro": "<|ro|>",  # Romanian
        "sk": "<|sk|>",  # Slovak
        "sl": "<|sl|>",  # Slovene
        "es": "<|es|>",  # Spanish
        "sv": "<|sv|>",  # Swedish
        "sq": "<|sq|>",  # Albanian
        # other languages
        "ast": "<|ast|>",  # Asturian
        "eu": "<|eu|>",  # Basque
        "br": "<|br|>",  # Breton
        "ca": "<|ca|>",  # Catalan
        "fy": "<|fy|>",  # Frisian
        "gl": "<|gl|>",  # Galician
        "oc": "<|oc|>",  # Occitan
        "rm": "<|rm|>",  # Romansh (Vallader, Sursilv)
        "sc": "<|sc|>",  # Sardinian
        "hsb": "<|hsb|>",  # Sorbian
        "cy": "<|cy|>",  # Welsh
    }

    task2token = {
        "transcribe": "<|transcribe|>",
        "translate": "<|translate|>",
        "summarize": "<|summarize|>",
        "reply": "<|reply|>",
    }

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls, audio_model_name_or_path, text_model_name_or_path, **kwargs
    ):
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            audio_model_name_or_path, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path, **kwargs)

        additional_tokens = [
            *list(cls.task2token.values()),
            *list(cls.lang2token.values()),
        ]

        tokenizer.add_tokens(additional_tokens, special_tokens=True)
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # def _add_lang_task_tokens(
    #     self,
    #     text_inputs: Dict[str, List],
    #     target_langs: List[str],
    #     target_tasks: List[str],
    # ):
    #     text_inputs["input_ids"] = [
    #         [ti[0]]  # bos_token
    #         + [
    #             self.tokenizer.convert_tokens_to_ids(self.lang2token[tl]),
    #             self.tokenizer.convert_tokens_to_ids(self.task2token[tt]),
    #         ]
    #         + ti[1:]
    #         for ti, tl, tt in zip(text_inputs["input_ids"], target_langs, target_tasks)
    #     ]
    #     text_inputs["attention_mask"] = [
    #         [1, 1] + am for am in text_inputs["attention_mask"]
    #     ]
    #     return text_inputs
    def _have_same_length(items: List):
        return all(len(item) == len(items[0]) for item in items)
    
    def _build_preamble_block(self, target_lang, target_task, text_preamble: Optional[List[str]]= None):
        if not text_preamble:
            preamble_block = [
                f"{self.tokenizer.bos_token} {self.lang2token[tl]} {self.task2token[tt]} {tp}"
                for tl, tt, tp in zip(target_lang, target_task, text_preamble)
            ]
        else:
            preamble_block = [
                f"{self.tokenizer.bos_token} {self.lang2token[tl]} {self.task2token[tt]}"
                for tl, tt in zip(target_lang, target_task)
            ]
        preamble_inputs = self.tokenizer(
            preamble_block,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True
        )
        return preamble_inputs

    def __call__(
        self,
        audio: AudioInput,
        task: Union[str, List[str]],
        target_lang: Union[str, List[str]],
        text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = None,
        text_preamble: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        """
        TODO: update docstring
        """
        if audio is None and text is None:
            raise ValueError(
                "You need to specify either an `audio` or `text` input to process."
            )
        
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            warnings.warn(
                "We currently support `return_tensors='pt'`. Setting it to `pt` for now.",
            )
        kwargs["return_tensors"] = "pt"
        
        output_kwargs = self._merge_kwargs(
            SpeechLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # 1. build audio component of the input
        # TODO: for now, audio is mandatory
        audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        items_count = audio_inputs["input_features"].shape[0]

        # 2. audio is followed by a block containing ids for task, target_lang, and optionally text preamble
        target_lang = target_lang if isinstance(target_lang, list) else [target_lang] * items_count
        task = task if isinstance(task, list) else [task] * items_count
        len_to_check = [target_lang, task]
        if text_preamble is not None:
            text_preamble = text_preamble if isinstance(text_preamble, list) else [text_preamble] * items_count
            len_to_check.append(text_preamble)
        if not self._have_same_length(len_to_check):
            raise ValueError("`lang`, `task`, and `text_preamble` must have the same length")
        
        preamble_inputs = self._build_preamble_block(target_lang, task, text_preamble)
        
        # 3. encode text if needed
        text_inputs = preamble_inputs
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            if not self._have_same_length([text, target_lang]):
                raise ValueError("`text` and `lang` must have the same length")

            tokenized_text = self.tokenizer(text, **output_kwargs["text_kwargs"])
            text_inputs = {
                "input_ids": torch.cat([preamble_inputs["input_ids"], tokenized_text["input_ids"]], dim=1),
                "attention_mask": torch.cat([preamble_inputs["input_ids"], tokenized_text["attention_mask"]], dim=1),
            }

        output_dict = {
            **{f"audio_{k}": v for k, v in audio_inputs.items()},
            **text_inputs,
        }
        if kwargs.get("return_labels", False):
            output_dict["labels"] = tokenized_text["input_ids"]

        return output_dict 

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
