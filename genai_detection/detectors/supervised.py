# Copyright 2024 Janek Bevendorff, Webis
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

import typing as t

import numpy as np
import torch
from torch.nn import functional as F

from genai_detection.detectors.detector_base import DetectorBase
from genai_detection.util import load_model, TorchDeviceMapType

__all__ = ['SupervisedDetector']


class SupervisedDetector(DetectorBase):
    """
    Supervised LLM text classifier.

    Requires a previously trained classification model. Use the ``finetune train-seq-cls`` or
    ``finetune tf-idf`` CLI commands to fit and save a suitable model.
    """

    def __init__(self,
                 hf_model=None,
                 sklearn_model=None,
                 device_map: TorchDeviceMapType = 'auto',
                 max_seq_length=800,
                 flash_attn=False,
                 quantization_bits=None,
                 **model_args):
        """
        :param hf_model: Huggingface name of path to fine-tuned transformer sequence classification model
        :param sklearn_model: path to trained and serialized scikit-learn model
        :param device_map: Torch device map for (hf_model only)
        :param max_seq_length: cut input texts to this manu tokens (hf_model only)
        :param flash_attn: Use Flash Attention 2.0 (hf_model only)
        :param quantization_bits: quantize sequence classification model to 4 or 8 bits (hf_model only)
        :param model_args: additional arguments to pass to model loader (hf_model only)
        """
        self.hf_model = None
        self.sklearn_clf = None
        self.max_seq_length = max_seq_length

        if hf_model and sklearn_model:
            raise ValueError('Cannot specify both hf_model and sklearn_model')
        elif hf_model:
            with torch.inference_mode():
                self.hf_model, self.tokenizer = load_model(
                    hf_model,
                    task_type='SEQ_CLS',
                    device_map=device_map,
                    flash_attn=flash_attn,
                    quantization_bits=quantization_bits,
                    tokenizer_max_length=max_seq_length,
                    **model_args)
        elif sklearn_model:
            from skops.io import load
            self.sklearn_clf, self.vec, self.pred_fn = load(
                sklearn_model, trusted=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']).values()
        else:
            raise ValueError('Must provide either hf_model or sklearn_model')

    def _predict(self, text: t.Iterable[str]) -> t.Union[torch.Tensor, np.ndarray]:
        if self.hf_model:
            with torch.inference_mode():
                tokens = self.tokenizer(text,
                                        return_tensors='pt',
                                        padding=True,
                                        max_length=self.max_seq_length,
                                        truncation='longest_first').to(self.hf_model.device)
                return self.hf_model(**tokens).logits.float().cpu()

        if self.sklearn_clf:
            return getattr(self.sklearn_clf, self.pred_fn)(self.vec.transform(text))

        raise ValueError('No model loaded.')

    def _normalize_scores(self, scores):
        if self.hf_model:
            return F.softmax(scores, dim=1)
        return scores

    def _get_score_impl(self, text):
        return self._predict(text)[:, 1]

    def _predict_impl(self, text):
        preds = self._predict(text)
        return preds.argmax(dim=1) if type(preds) is torch.Tensor else np.argmax(preds, axis=1)

    def _predict_with_score_impl(self, text):
        scores = self._predict(text)
        preds = scores.argmax(dim=1) if type(scores) is torch.Tensor else np.argmax(scores, axis=1)
        return preds, scores[:, 1]
