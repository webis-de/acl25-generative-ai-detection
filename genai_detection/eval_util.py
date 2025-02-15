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

import typing as t

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


__all__ = [
    'compute_metrics',
]


def compute_metrics(logits: np.ndarray, labels: t.Union[t.List[int], np.array],
                    scores=None, machine_label=1) -> t.Dict[str, t.Any]:
    """
    Compute evaluation metrics given a model's logits / output probabilities and the true labels.

    :param logits: logits / probabilities or predicted labels, shape = (batch_size, 2) or shape = (batch_size,)
    :param labels: true labels, shape = (batch_size,)
    :param scores: list of arbitrarily scaled machine-likeness scores (higher -> more machine-like), can
                   be supplied for AUC-ROC calculation if ``logits`` are discrete classes
    :param machine_label: which of the two labels is the machine label
    :return: computed metrics as dict
    """
    h, m = 1 - machine_label, machine_label
    logits_argmax = logits if len(logits.shape) == 1 else logits.argmax(axis=1)
    cm = confusion_matrix(labels, logits_argmax, labels=[h, m])
    if scores is not None:
        auc_roc = roc_auc_score(labels, scores)
    elif len(logits.shape) == 2:
        auc_roc = roc_auc_score(labels, logits[:, m])
    else:
        auc_roc = roc_auc_score(labels, logits_argmax)

    return {
        'acc': accuracy_score(labels, logits_argmax),
        'auc_roc': auc_roc,
        'confusion': cm.tolist(),
        'fpr': cm[h, m] / cm[h].sum(),
        'fnr': cm[m, h] / cm[m].sum(),
        'human_prec': precision_score(labels, logits_argmax, pos_label=h, zero_division=0.0),
        'human_rec': recall_score(labels, logits_argmax, pos_label=h, zero_division=0.0),
        'human_f1': f1_score(labels, logits_argmax, pos_label=h, zero_division=0.0),
        'machine_prec': precision_score(labels, logits_argmax, pos_label=m, zero_division=0.0),
        'machine_rec': recall_score(labels, logits_argmax, pos_label=m, zero_division=0.0),
        'machine_f1': f1_score(labels, logits_argmax, pos_label=m, zero_division=0.0),
    }
