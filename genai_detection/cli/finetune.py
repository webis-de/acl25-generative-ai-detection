import json
import os
from pathlib import Path
import random
import typing as t

from accelerate import Accelerator
import click
from datasets import DatasetDict
import numpy as np
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm
from transformers import DataCollatorWithPadding, set_seed, Trainer, TrainingArguments
from xgboost import XGBClassifier

from genai_detection.util import load_model
from genai_detection.eval_util import compute_metrics
from genai_detection.cli.dataset import DATASET_FEATURES, create_dataset_split


_SEQ_CLS_MODEL_EXTRA_ARGS = dict(
    num_labels=DATASET_FEATURES['label'].num_classes,
    label2id={n: DATASET_FEATURES['label'].str2int(n) for n in DATASET_FEATURES['label'].names},
    id2label={DATASET_FEATURES['label'].str2int(n): n for n in DATASET_FEATURES['label'].names},
)


class WeightedTrainer(Trainer):
    def __init__(self, model, class_weights: t.Iterable[float], *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        weights = torch.tensor(self.class_weights, device=model.device, dtype=logits.dtype)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def get_class_weights(labels, max_sample_size=100_000):
        """
        Estimate class weights from binary labels.
        
        :param labels: dataset labels as list of tensor
        :param max_sample_size: maximum number of samples to read from ``labels``
        :return: weights for negative and positive class
        """
        s = labels[:max_sample_size]
        pos_count = sum(s)
        neg_count = len(s) - pos_count
        w = np.array([1. / neg_count, 1. / pos_count])
        return (w / np.max(w)).tolist()


def map_dataset_to_tensors(dataset, tokenizer, max_seq_len=1024, batched=True, batch_size=1000, **tokenizer_kwargs):
    def _token_fn(ds_slice):
        return {
            **tokenizer(ds_slice['text'], truncation=True, max_length=max_seq_len, **tokenizer_kwargs),
            'labels': ds_slice['label']
        }

    drop_cols = [n for n in dataset.column_names if n not in ['labels', 'input_ids', 'attention_mask']]
    dataset = dataset.map(_token_fn,
                          batched=batched,
                          batch_size=batch_size,
                          num_proc=len(os.sched_getaffinity(0)) - 1).remove_columns(drop_cols)
    dataset.set_format('torch')
    return dataset


@click.group()
def main():
    pass


@main.command()
@click.argument('basemodel')
@click.argument('dataset', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, file_okay=False), default='data/model',
              help='Output path for trained model')
@click.option('-r', '--resume', is_flag=True, help='Resume from previous checkpoint')
@click.option('-f', '--flash-attn', is_flag=True, help='Use Flash Attention')
@click.option('-q', '--quantize', type=click.Choice(['4', '8']), help='Quantize model')
@click.option('-v', '--validation-size', type=click.FloatRange(0, 1),
              help='Validation split size (set to zero to use fixed val split from dataset)')
@click.option('-n', '--max-steps', type=click.IntRange(0), default=10_000,
              help='Max number of training steps')
@click.option('-l', '--learning-rate', type=click.FloatRange(0, min_open=True), default=1e-4)
@click.option('-m', '--max-seq-len', type=click.IntRange(1), default=800,
              help='Maximum length to pad or truncate training sequences to')
@click.option('-c', '--save-steps', type=click.IntRange(1), default=250,
              help='Training steps after which to save a checkpoint')
@click.option('--gradient-checkpointing', is_flag=True, help='Use gradient checkpointing')
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
def train_seq_cls(basemodel, dataset, output, resume, flash_attn, quantize, validation_size,
                  max_steps, learning_rate, max_seq_len, save_steps, gradient_checkpointing, seed):
    """
    Fine-tune an LLM for sequence classification on a given dataset.
    """
    if max_steps % save_steps != 0:
        raise click.UsageError('--max-steps must be a multiple of --save-steps.')

    set_seed(seed)

    ds_dict = DatasetDict.load_from_disk(dataset)
    ds_train, ds_val = create_dataset_split(ds_dict['train'],
                                            test_split_size=validation_size,
                                            shuffle=True,
                                            seed=seed)
    if validation_size is None:
        ds_val = ds_dict.get('validation')

    (model, load_info), tokenizer = load_model(basemodel,
                                               task_type='SEQ_CLS',
                                               device_map=Accelerator().process_index,
                                               flash_attn=flash_attn,
                                               quantization_bits=quantize,
                                               output_loading_info=True,
                                               **_SEQ_CLS_MODEL_EXTRA_ARGS)
    print('Training data:', ds_train)
    if ds_val:
        print('Validation data:', ds_val)

    if quantize:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias='none',
        # target_modules=['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj', 'o_proj', 'gate_proj'],
        target_modules='all-linear',
        use_rslora=True,
    )
    model = get_peft_model(model, peft_config)

    trainer = WeightedTrainer(
        model,
        WeightedTrainer.get_class_weights(ds_train['label']),
        train_dataset=map_dataset_to_tensors(ds_train, tokenizer, max_seq_len),
        eval_dataset=map_dataset_to_tensors(ds_val, tokenizer, max_seq_len) if ds_val else None,
        data_collator=DataCollatorWithPadding(tokenizer, padding='longest'),
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(*p, machine_label=DATASET_FEATURES['label'].str2int('machine')),
        args=TrainingArguments(
            auto_find_batch_size=True,
            ddp_find_unused_parameters=False,
            learning_rate=learning_rate,
            output_dir=output,
            max_steps=max_steps,
            save_steps=save_steps,
            eval_strategy='steps' if ds_val else 'no',
            eval_steps=save_steps,
            logging_steps=max(1, save_steps // 10),
            log_level='info',
            lr_scheduler_type='cosine_with_restarts',
            lr_scheduler_kwargs={'num_cycles': max_steps // save_steps},
            warmup_steps=0,
            warmup_ratio=0.0,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={'use_reentrant': False} if gradient_checkpointing else None,
            group_by_length=True,
            bf16=True,
            seed=seed,
        )
    )
    model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=resume)


@main.command()
@click.argument('model')
@click.argument('dataset', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, file_okay=False, writable=True),
              default='data/eval', help='Output directory for predictions')
@click.option('-s', '--split-name', type=click.Choice(['train', 'validation', 'test']), default='test',
              help='Named split to load from dataset')
@click.option('-b', '--batch-size', type=click.IntRange(1), default=1, help='Inference batch size')
@click.option('-f', '--flash-attn', is_flag=True, help='Use Flash Attention')
@click.option('-q', '--quantize', type=click.Choice(['4', '8']), help='Quantize model')
@click.option('-m', '--max-seq-len', type=click.IntRange(1), default=800,
              help='Maximum length to pad or truncate training sequences to')
def eval_seq_cls(model, dataset, output, split_name, batch_size, flash_attn, quantize, max_seq_len):
    """
    Evaluate fine-tuned sequence classification model.
    """

    with torch.inference_mode():
        ds_test = DatasetDict.load_from_disk(dataset)[split_name]
        model, tokenizer = load_model(model,
                                      task_type='SEQ_CLS',
                                      device_map='auto',
                                      flash_attn=flash_attn,
                                      quantization_bits=quantize,
                                      **_SEQ_CLS_MODEL_EXTRA_ARGS)
        model.eval()

        logits = []
        for batch in tqdm(ds_test.batch(batch_size), desc='Inferring batch'):
            tokens = tokenizer(batch['text'],
                               return_tensors='pt',
                               padding=True,
                               max_length=max_seq_len,
                               truncation='longest_first').to(model.device)
            logits.extend(model(**tokens).logits.float().tolist())

    logits = np.array(logits)
    ds_test = (ds_test
               .select_columns(['id', 'label'])
               .add_column('human_logits', logits[:, ds_test.features['label'].str2int('human')])
               .add_column('machine_logits', logits[:, ds_test.features['label'].str2int('machine')])
               .add_column('pred_label', logits.argmax(axis=1))
               .rename_column('label', 'true_label'))

    # noinspection DuplicatedCode
    output = Path(output)
    ds_name = Path(dataset).name
    output.mkdir(parents=True, exist_ok=True)
    ds_test.to_csv(output / f'{ds_name}-{split_name}-predictions.csv')
    eval_str = json.dumps(compute_metrics(logits, ds_test['true_label'],
                                          machine_label=DATASET_FEATURES['label'].str2int('machine')), indent=2)
    open(output / f'{ds_name}-{split_name}-eval.json', 'w').write(eval_str)
    print(eval_str)


@main.command()
@click.argument('dataset', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, file_okay=False), default='data/model-tfidf',
              help='Output path for trained model')
@click.option('-t', '--model-type', type=click.Choice(['svm', 'logreg', 'xgboost']),
              help='Model type to train', default='svm')
@click.option('-m', '--max-features', type=click.IntRange(1),
              help='Maximum number of TF-IDF features', default=1000)
@click.option('--ngram-min', type=click.IntRange(1), help='Minimum ngram length', default=1)
@click.option('--ngram-max', type=click.IntRange(1), help='Maximum ngram length', default=1)
@click.option('-v', '--validation-size', type=click.FloatRange(0, 1),
              help='Validation split size (set to zero to use fixed val split from dataset)')
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
def train_tf_idf(dataset, output, model_type, max_features, ngram_min, ngram_max, validation_size, seed):
    """
    Train a supervised TF-IDF-based LLM detection model.
    """
    np.random.seed(seed)
    random.seed(seed)
    ds_dict = DatasetDict.load_from_disk(dataset)
    ds_train, ds_val = create_dataset_split(ds_dict['train'],
                                            test_split_size=validation_size,
                                            shuffle=True,
                                            seed=seed)
    if validation_size is None:
        ds_val = ds_dict.get('validation')

    class_weights = WeightedTrainer.get_class_weights(ds_train['label'])
    sample_weights = [class_weights[l] for l in ds_train['label']]

    pred_fn = 'predict_proba'
    if model_type == 'svm':
        clf = LinearSVC()
        pred_fn = '_predict_proba_lr'
    elif model_type == 'logreg':
        clf = LogisticRegression()
    elif model_type == 'xgboost':
        clf = XGBClassifier()
    else:
        raise ValueError('Invalid model type')

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(ngram_min, ngram_max))
    clf.fit(vec.fit_transform(ds_train['text']), ds_train['label'], sample_weight=sample_weights)
    if ds_val is not None:
        probas = getattr(clf, pred_fn)(vec.transform(ds_val['text']))
        print(json.dumps(compute_metrics(probas, ds_val['label'],
                                         machine_label=DATASET_FEATURES['label'].str2int('machine')), indent=2))

    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path /= f'{Path(dataset).name}-tfidf-{model_type}.skops'

    from skops.io import dump
    dump({'clf': clf, 'vec': vec, 'pred_fn': pred_fn}, str(out_path))


@main.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.argument('dataset', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, file_okay=False, writable=True),
              default='data/eval', help='Output directory for predictions')
@click.option('-s', '--split-name', type=click.Choice(['train', 'validation', 'test']), default='test',
              help='Named split to load from dataset')
def eval_tf_idf(model, dataset, output, split_name):
    """
    Evaluate a trained TF-IDF-based LLM detection model.
    """

    from skops.io import load
    ds_test = DatasetDict.load_from_disk(dataset)[split_name]
    clf, vec, pred_fn = load(model, trusted=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']).values()

    probas = getattr(clf, pred_fn)(vec.transform(ds_test['text']))
    ds_test = (ds_test
               .select_columns(['id', 'label'])
               .add_column('human_probas', probas[:, ds_test.features['label'].str2int('human')])
               .add_column('machine_probas', probas[:, ds_test.features['label'].str2int('machine')])
               .add_column('pred_label', probas.argmax(axis=1))
               .rename_column('label', 'true_label'))

    # noinspection DuplicatedCode
    output = Path(output)
    ds_name = Path(dataset).name
    output.mkdir(parents=True, exist_ok=True)
    ds_test.to_csv(output / f'{ds_name}-{split_name}-predictions.csv')
    eval_str = json.dumps(compute_metrics(probas, ds_test['true_label'],
                                          machine_label=DATASET_FEATURES['label'].str2int('machine')), indent=2)
    open(output / f'{ds_name}-{split_name}-eval.json', 'w').write(eval_str)
    print(eval_str)
