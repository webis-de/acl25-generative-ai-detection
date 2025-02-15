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

import re
import typing as t


from genai_detection.detectors.detector_base import DetectorBase

__all__ = ['WordLengthDetector']

_WS_RE = re.compile(r'\s+')


class WordLengthDetector(DetectorBase):
    """
    Baseline LLM detector that distinguishes LLM and human texts based on their average word lengths.
    """

    def __init__(self, word_length=5.1, shorter_is_ai=False):
        """
        :param word_length: word length threshold
        :param shorter_is_ai: whether shorter texts are classified as AI (default: longer texts are AI)
        """
        self.word_length = word_length
        self.shorter_is_ai = shorter_is_ai

    def _get_score_impl(self, text: t.Iterable[str]) -> t.Iterable[float]:
        for t in text:
            word_lens = [len(w) for w in _WS_RE.split(t.strip())]
            avg_len = sum(word_lens) / len(word_lens)
            yield -avg_len if self.shorter_is_ai else avg_len

    def _threshold(self, scores: t.Iterable[float]) -> t.Iterable[bool]:
        thresh = self.word_length
        if self.shorter_is_ai:
            thresh *= -1
        yield from (s >= thresh for s in scores)

    def _predict_impl(self, text: t.Iterable[str]) -> t.Iterable[bool]:
        yield from self._threshold(self._get_score_impl(text))

    def _predict_with_score_impl(self, text: t.Iterable[str]) -> t.Tuple[t.List[bool], t.List[float]]:
        scores = list(self._get_score_impl(text))
        preds = list(self._threshold(scores))
        return preds, scores
