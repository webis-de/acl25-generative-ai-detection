# Copyright 2025 Janek Bevendorff, Webis
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

from collections import defaultdict
from itertools import batched
from random import randint
import re
from typing import Iterable, List
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from genai_detection.detectors.detector_base import DetectorBase

__all__ = ['UnmaskingDetector']

warnings.simplefilter('ignore', category=ConvergenceWarning)


class UnmaskingDetector(DetectorBase):
    """
    Baseline LLM detector calculating normalized cumulative sum of the authorship unmasking curve points.

    The input is a list of texts where text ``i`` and text ``i+1`` belong to a pair. The output is a list
    of curve points for each pair that is half the size of the input list.

    References:
    ===========
        Koppel, Moshe, and Jonathan Schler. 2004. “Authorship Verification as a One-Class
        Classification Problem.” In Proceedings, Twenty-First International Conference on
        Machine Learning, ICML 2004, 489–95.

        Bevendorff, Janek, Benno Stein, Matthias Hagen, and Martin Potthast. 2019. “Generalizing
        Unmasking for Short Texts.” In Proceedings of the 2019 Conference of the North, 654–59.
        Stroudsburg, PA, USA: Association for Computational Linguistics.
    """
    def __init__(self, rounds=30, top_n=250, cv_folds=10, n_delete=3, tokenizer=None, shared_vocab_only=False,
                 chunk_size=500, relative_freqs=True, bootstrap=True, n_chunks=60, smooth=None, strict=False):
        """
        :param rounds: number of deconstruction rounds
        :param top_n: number of top tokens to sample
        :param cv_folds: number of cross-validation folds
        :param n_delete: number of features to eliminate in each round
        :param tokenizer: custom tokenizer function (must accept exactly one parameter, defaults to character trigrams)
        :param shared_vocab_only: restrict analysis to shared vocabulary
        :param chunk_size: size of bootstrapped chunks
        :param relative_freqs: use relative term frequencies
        :param bootstrap: bootstrap chunks by oversampling tokens (allows the user of shorter texts)
        :param n_chunks: number of chunks to generate (only if ``bootstrap=True``)
        :param smooth: kernel size for smoothing curves
        :param strict: throw a :class:`ValueError` if the last input batch is not a full pair
        """

        if top_n <= 2 * n_delete * rounds:
            raise ValueError('top_n must be larger than 2 * n_delete * n_rounds.')

        self.rounds = rounds
        self.top_n = top_n
        self.shared_vocab_only = shared_vocab_only
        self.cv_folds = cv_folds
        self.n_delete = n_delete
        self.chunk_size = chunk_size
        self.relative_freqs = relative_freqs
        self.bootstrap = bootstrap
        self.n_chunks = n_chunks
        self.smoothing_kernel_size = smooth
        self.strict = strict
        self.tokenizer = tokenizer or self.tokenize_char_ngrams

    def get_curves(self, text: Iterable[str]) -> List[np.ndarray]:
        """
        Get unmasking accuracy curves for pairs of input texts.

        :param text: list of text pairs where text ``i`` and text ``i+1`` always belong to a pair
        :return: list of curve points (half the size of the input)
        """
        curves = []
        for t in batched(text, 2):
            if len(t) != 2:
                if self.strict:
                    raise ValueError('Final batch is not a full pair.')
                break
            tokens_left = self.tokenizer(t[0])
            tokens_right = self.tokenizer(t[1])

            chunks_left = self.create_chunks(tokens_left, self.chunk_size, self.bootstrap, self.n_chunks)
            chunks_right = self.create_chunks(tokens_right, self.chunk_size, self.bootstrap, self.n_chunks)

            freqs_left = self.get_token_freqs(tokens_left)
            freqs_right = self.get_token_freqs(tokens_right)
            if self.shared_vocab_only:
                shared_tokens = freqs_left.keys() & freqs_right.keys()
            else:
                shared_tokens = freqs_left.keys() | freqs_right.keys()
            top_tokens = sorted(shared_tokens, key=lambda x: freqs_left[x] + freqs_right[x], reverse=True)[:self.top_n]

            x_left = self.chunks_to_matrix(chunks_left, top_tokens)
            x_right = self.chunks_to_matrix(chunks_right, top_tokens)

            curves.append(self.deconstruct(x_left, x_right,
                                           self.rounds,
                                           self.n_delete,
                                           self.cv_folds,
                                           self.smoothing_kernel_size))
        return curves

    def _get_score_impl(self, text: Iterable[str]) -> List[float]:
        scores = []
        for degen_acc in self.get_curves(text):
            score = 2 * np.sum(degen_acc - 0.75) / len(degen_acc)
            scores.append(1.0 / (1.0 + np.exp(score)))
        return scores

    @staticmethod
    def tokenize_char_ngrams(text, n=3, normalize_ws=True):
        """
        Tokenize input text into character n-grams.

        :param text: input text
        :param n: n-gram order
        :param normalize_ws: collapse whitespace before tokenization
        :return: list of n-gram tokens
        """
        text = text.strip()
        if normalize_ws:
            text = re.sub(r'\s+', ' ', text)
        return [text[i:i + n] for i in range(0, len(text) - n + 1)]

    @staticmethod
    def tokenize_word_ngrams(text, n=3, word_tokenizer=None):
        """
        Tokenize input text into word n-grams.

        :param text: input text
        :param n: n-gram order
        :param word_tokenizer: word tokenizer to use (defaults to :meth:`tokenize_words`)
        :return: list of word n-gram tokens
        """
        tokens = word_tokenizer(text.strip()) if word_tokenizer else re.findall(r'\w+', text.strip())
        return [' '.join(tokens[i:i + n]) for i in range(0, len(tokens) - n + 1)]

    @staticmethod
    def tokenize_words(text):
        """
        Tokenize text into regex ``\\w+`` "word" tokens.

        :param text: input text
        :return: list of word tokens
        """
        return re.findall(r'\w+', text)

    @staticmethod
    def tokenize_whitespace(text):
        """
        Tokenize text by whitespace.

        :param text: input text
        :return: list of word tokens
        """
        return text.split()

    def get_token_freqs(self, *token_lists):
        """Get combined frequency dictionary for all tokens in the input sequence(s)."""
        freqs = defaultdict(int)
        n = 0
        for tokens in token_lists:
            for t in tokens:
                freqs[t] += 1
                n += 1
        if self.relative_freqs:
            for f in freqs:
                freqs[f] /= n
        return freqs

    @staticmethod
    def bootstrap_tokens(tokens, n_tokens):
        """
        Sample tokens from the input list proportionally with replacement.

        :param tokens: sequence of tokens
        :param n_tokens: number of tokens to sample from input sequence
        :return: list of sampled tokens
        """
        return [tokens[randint(0, len(tokens) - 1)] for _ in range(n_tokens)]

    @classmethod
    def create_chunks(cls, tokens, chunk_size, bootstrap=False, n_chunks=None):
        """
        Create chunks of tokens from the input token sequence, with or without bootstrapping.

        :param tokens: sequence of tokens
        :param chunk_size: size of chunks to generate (output chunks will be at least half the size
                           if ``bootstrap=False``).
        :param bootstrap: if ``True``, (over-)sample tokens from input sequence, otherwise just cut input into pieces
        :param n_chunks: number of chunks to generate if ``bootstrap=True``, otherwise determined
                         by input length and ``chunk_size``
        :return: list of chunks
        """
        if bootstrap:
            return [cls.bootstrap_tokens(tokens, chunk_size) for _ in range(n_chunks)]

        return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)
                if len(tokens) - i >= chunk_size / 2]

    def chunks_to_matrix(self, chunks, top_token_list):
        """
        Transform list of chunks into matrix of term frequencies of the top tokens.

        :param chunks: list of input chunks
        :param top_token_list: list of top tokens to include in the matrix
        :return: Numpy array of term frequencies, `shape = (len(chunks), len(top_token_list))`
        """
        mat = []
        for c in chunks:
            freq = self.get_token_freqs(c)
            mat.append([freq[t] for t in top_token_list])

        return np.array(mat)

    @staticmethod
    def deconstruct(x_left, x_right, rounds, n_delete, cv_folds=10, smoothing_kernel_size=None):
        """
        Iteratively classify and deconstruct a text pair representation and return the resulting accuracy curve.

        :param x_left: "left" text chunk representation matrix
        :param x_right: "right" text chunk representation matrix
        :param rounds: number of deconstruction rounds
        :param n_delete: number of positive and negative features to eliminate in each round
        :param cv_folds: number of cross-validation folds
        :param smoothing_kernel_size: curve smoothing kernel size
        :return: list of classification accuracy values
        """
        X = np.vstack((x_left, x_right))
        y = np.zeros(len(x_left) + len(x_right))
        y[len(x_left):] = 1.0
        X, y = shuffle(X, y)

        rounds = min(rounds, (X.shape[1] - 1) // n_delete)
        scores = np.zeros(rounds)
        for i in range(rounds):
            if X.shape[1] == 0:
                warnings.warn('Feature dimension reduced to zero. Either increase top_n or reduce n_delete.')
                break
            cv = cross_validate(LinearSVC(dual='auto'), X, y, cv=cv_folds, return_estimator=True)
            scores[i] = cv['test_score'].mean()
            coefs = np.mean(np.vstack([c.coef_.squeeze() for c in cv['estimator']]), axis=0)
            argsort = np.argsort(coefs)
            coefs_sorted = coefs[argsort]
            top_arg = np.concat([argsort[coefs_sorted > 0][-n_delete:],
                                 argsort[coefs_sorted < 0][:n_delete]])
            X[:, top_arg] = 0

        if smoothing_kernel_size:
            scores = np.convolve(scores, np.ones(smoothing_kernel_size) / smoothing_kernel_size, mode='valid')

        return scores
