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
from genai_detection.detectors.detector_base import DetectorBase

__all__ = ['MajorityEnsemble']


class MajorityEnsemble(DetectorBase):
    """
    LLM detector using the majority decisions of other detectors.
    """

    def __init__(self, detectors: t.List[DetectorBase], detector_weights=None):
        """
        :param detectors: instance of other detectors to use
        :param detector_weights: weights for each detector (equal weights by default)
        """
        if len(detectors) < 2:
            raise ValueError('Need at least 2 detectors.')

        self.detectors = detectors

        if detector_weights:
            if len(detector_weights) != len(detectors):
                raise ValueError('Detector weight shape must match number of detectors.')
            self.detector_weights = np.array(detector_weights) / np.sum(detector_weights)
        else:
            self.detector_weights = np.ones(len(detectors)) / len(detectors)

    def _get_score_impl(self, text):
        return np.sum([d.predict(text) * self.detector_weights[i]
                       for i, d in enumerate(self.detectors)], axis=0)

    def _predict_impl(self, text):
        return self._get_score_impl(text) > .5

    def _predict_with_score_impl(self, text):
        scores = self._get_score_impl(text)
        return (scores > .5), scores
