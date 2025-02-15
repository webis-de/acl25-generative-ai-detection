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

from pathlib import Path

import click
import json

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from genai_detection.eval_util import compute_metrics


@click.group()
def main():
    pass


def load_input(input_name, input_split_name=None):
    import datasets

    path = Path(input_name)
    if path.is_file() and path.name.endswith('.csv'):
        df = pd.read_csv(input_name)
        if len(df.columns) == 1:
            df.columns = ['text']
            df.index.name = 'id'
        elif len(df.columns) == 2:
            df.columns = ['text', 'label']
            df.index.name = 'id'
        elif len(df.columns) == 3:
            df.columns = ['id', 'text', 'label']
            df.set_index('id')
        else:
            raise click.UsageError('Invalid CSV schema. Please convert the dataset first.')
        return datasets.Dataset.from_pandas(df, preserve_index=True)

    if path.is_file() and path.name.endswith('.txt'):
        return datasets.Dataset.from_dict({
            'id': path.name,
            'text': path.read_text()
        })

    if path.is_dir():
        if (path / 'dataset_dict.json').is_file():
            if not input_split_name:
                raise click.UsageError('Input seems to be a DatasetDict, but no split name was given.')
            return datasets.load_from_disk(str(path))[input_split_name]

        if (path / 'dataset_info.json').is_file():
            return datasets.load_from_disk(str(path))

        g = list(path.glob('*.txt'))
        if len(g) > 0:
            return datasets.Dataset.from_dict({
                'id': [f.name for f in g],
                'text': [f.read_text() for f in g]
            })

    raise click.UsageError('Unknown input data format')


def detect(detector, dataset, output_file, split_name='test', batch_size=1):
    """
    Run a detector on an input dataset and output predictions

    :param detector: DetectorBase
    :param dataset: input dataset
    :param split_name: input dataset split
    :param output_file: output filename
    :param batch_size: prediction batch size
    """

    dataset: Dataset = load_input(dataset, split_name)
    preds = []
    scores = []
    for batch in tqdm(dataset.select_columns('text').batch(batch_size), desc='Making predictions', unit=' batches'):
        p, s = detector.predict_with_score(batch['text'])
        preds.extend(p)
        scores.extend(s)

    out_df = pd.concat([
        pd.Series(dataset['id']),
        pd.Series(preds),
        pd.Series(scores),
    ], axis=1)
    out_df.columns = ['id', 'pred_label', 'score']

    json_file = None
    if 'label' in dataset.column_names:
        out_df.insert(1, 'true_label', pd.Series(dataset['label']))
        eval_str = json.dumps(compute_metrics(out_df['pred_label'], out_df['true_label'], scores=scores), indent=2)
        json_file = Path(output_file.rsplit('.', 1)[0] + '-eval.json')
        open(json_file, 'w').write(eval_str)
        print('Evaluation:\n', eval_str)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)
    print(f'Predictions written to {output_file}.')
    if json_file:
        print(f'Evaluation written to {json_file}.')


class DetectorCommand(click.Command):
    """
    Detector command class with pre-defined shared CLI arguments.
    """
    def __init__(self, *args, with_model_path=False, with_gpu_opts=False, **kwargs):
        super().__init__(*args, **kwargs)

        gpu_opts = [
            click.core.Option(('-q', '--quantize'),
                              type=click.Choice(['4', '8']),
                              help='Quantize model weights on the GPU'),
            click.core.Option(('-f', '--flash-attn'),
                              is_flag=True,
                              help='Use Flash Attention 2.0 (requires Ampere GPU or better)')
        ] if with_gpu_opts else []

        model_path_opt = [
            click.core.Argument(('model',), type=click.Path(exists=True))
        ] if with_model_path else []

        self.params = [
            *model_path_opt,
            click.core.Argument(('input_dataset',), type=click.Path(file_okay=False, exists=True)),
            click.core.Option(('-o', '--output-csv',),
                              type=click.Path(file_okay=True, exists=False),
                              help='Path for CSV output file with predictions',
                              default='data/predictions.csv'),
            click.core.Option(('-s', '--dataset-split'),
                              type=click.Choice(['train', 'validation', 'test']),
                              help='Input dataset split',
                              default='test'),
            click.core.Option(('-b', '--batch-size'),
                              type=click.IntRange(1),
                              help='Batch size',
                              default=1),
            *gpu_opts,
            *self.params
        ]


@main.command(cls=DetectorCommand, with_gpu_opts=True)
@click.option('--observer', help='Observer model path or name', default='tiiuae/falcon-7b')
@click.option('--performer', help='Performer model path or name', default='tiiuae/falcon-7b-instruct')
@click.option('--device1', help='Observer model device', default='auto')
@click.option('--device2', help='Performer model device', default='auto')
def binoculars(input_dataset, output_csv, dataset_split, batch_size, quantize, flash_attn,
               observer, performer, device1, device2):
    """
    Binoculars zero-shot AI detector (Hans et al., 2024).

    \b
    References:
    ===========
        Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi,
        Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024.
        “Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated
        Text.” arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2401.12070.
    """
    from genai_detection.detectors.binoculars import Binoculars

    detector = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        device1=device1,
        device2=device2)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command(cls=DetectorCommand, with_gpu_opts=True)
@click.option('-m', '--scoring-mode', type=click.Choice(['lrr', 'npr']), default='lrr')
@click.option('-s', '--span-length', type=int, default=2, help='Size of mask token spans')
@click.option('-p', '--perturb-pct', type=click.FloatRange(0, 1), default=0.3,
              help='Percentage of tokens to perturb')
@click.option('-c', '--perturb-cache', type=click.Path(file_okay=False), help='Perturbation cache directory')
@click.option('-n', '--n-samples', type=int, default=20,
              help='Number of perturbed samples to generate for NPR')
@click.option('--base-model', help='Base detection model path or name', default='tiiuae/falcon-7b')
@click.option('--perturb-model', help='Perturbation model path or name for NPR', default='t5-large')
@click.option('--device1', help='Base model device', default='auto')
@click.option('--device2', help='Perturbation model device', default='auto')
def detectllm(input_dataset, output_csv, dataset_split, batch_size, quantize, flash_attn,
              scoring_mode, span_length, perturb_pct, perturb_cache, n_samples,
              base_model, perturb_model, device1, device2):
    """
    DetectLLM zero-shot AI detector (Su et al., 2023).

    \b
    References:
    ===========
        Su, Jinyan, Terry Yue Zhuo, Di Wang, and Preslav Nakov. 2023. “DetectLLM: Leveraging
        Log Rank Information for Zero-Shot Detection of Machine-Generated Text.”
        arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2306.05540.
    """
    from genai_detection.detectors.detectllm import DetectLLM
    from genai_detection.perturbators.t5_mask import T5MaskPerturbator

    perturbator = None
    if perturb_model and scoring_mode == 'npr':
        perturbator = T5MaskPerturbator(
            cache_dir=perturb_cache,
            model_name=perturb_model,
            quantization_bits=quantize,
            flash_attn=flash_attn,
            device=device2,
            span_length=span_length,
            mask_pct=perturb_pct,
            batch_size=batch_size)
    detector = DetectLLM(
        scoring_mode=scoring_mode,
        base_model=base_model,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        perturbator=perturbator,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device1)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command(cls=DetectorCommand, with_gpu_opts=True)
@click.option('-s', '--span-length', type=int, default=2, help='Size of mask token spans')
@click.option('-p', '--perturb-pct', type=click.FloatRange(0, 1), default=0.3,
              help='Percentage of tokens to perturb')
@click.option('-c', '--perturb-cache', type=click.Path(file_okay=False), help='Perturbation cache directory')
@click.option('-n', '--n-samples', type=int, default=20,
              help='Number of perturbed samples to generate')
@click.option('--base-model', help='Base detection model path or name', default='tiiuae/falcon-7b')
@click.option('--perturb-model', help='Perturbation model path or name', default='t5-large')
@click.option('--device1', help='Base model device', default='auto')
@click.option('--device2', help='Perturbation model device', default='auto')
def detectgpt(input_dataset, output_csv, dataset_split, batch_size, quantize, flash_attn,
              span_length, perturb_pct, perturb_cache, n_samples,
              base_model, perturb_model, device1, device2):
    """
    DetectGPT zero-shot AI detector (Mitchell et al., 2023).

    \b
    References:
    ===========
        Mitchell, Eric, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning,
        and Chelsea Finn. 2023. “DetectGPT: Zero-Shot Machine-Generated Text
        Detection Using Probability Curvature.” arXiv [Cs.CL]. arXiv.
        http://arxiv.org/abs/2301.11305.
    """
    from genai_detection.detectors.detectgpt import DetectGPT
    from genai_detection.perturbators.t5_mask import T5MaskPerturbator

    perturbator = T5MaskPerturbator(
        cache_dir=perturb_cache,
        model_name=perturb_model,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        device=device2,
        span_length=span_length,
        mask_pct=perturb_pct,
        batch_size=batch_size)
    detector = DetectGPT(
        base_model=base_model,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        perturbator=perturbator,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device1)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command()
@click.argument('input_dataset', type=click.Path(file_okay=False, exists=True))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('-s', '--dataset-split', type=click.Choice(['train', 'validation', 'test']),
              help='Input dataset split', default='test')
@click.option('-b', '--batch-size', type=int, default=20, help='GPU task batch size')
@click.option('-q', '--quantize', type=click.Choice(['4', '8']))
@click.option('-f', '--flash-attn', is_flag=True, help='Use flash-attn 2 (requires Ampere GPU or better)')
@click.option('--perturb-model', help='Perturbation model', default='t5-3b', show_default=True)
@click.option('--device', help='Perturb model device', default='auto', show_default=True)
@click.option('-s', '--span-length', type=int, default=2, show_default=True, help='Size of mask token spans')
@click.option('-p', '--perturb-pct', type=click.FloatRange(0, 1), default=0.3, show_default=True,
              help='Percentage of tokens to perturb')
@click.option('-n', '--n-samples', type=int, default=20, show_default=True,
              help='Number of perturbed samples to generate')
def detectgpt_cache(input_dataset, output_dir, dataset_split, batch_size, quantize, flash_attn,
                    perturb_model, device, span_length, perturb_pct, n_samples):
    """
    Generate and cache T5 mask perturbations for DetectGPT.
    """

    from genai_detection.perturbators.t5_mask import T5MaskPerturbator
    pert = T5MaskPerturbator(
        cache_dir=output_dir,
        model_name=perturb_model,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        device=device,
        span_length=span_length,
        mask_pct=perturb_pct,
        batch_size=batch_size)

    dataset = load_input(input_dataset, dataset_split)
    for batch in tqdm(dataset.batch(batch_size), desc='Perturbing input texts', unit=' batches'):
        pert.perturb(batch['text'], n_samples)


@main.command(cls=DetectorCommand, with_gpu_opts=True)
@click.option('-n', '--n-samples', type=int, default=10000,
              help='Number of perturbed samples to generate')
@click.option('--base-model', help='Base detection model path or name', default='tiiuae/falcon-7b')
@click.option('--device', help='Base model device', default='auto')
def fastdetectgpt(input_dataset, output_csv, dataset_split, batch_size, quantize, flash_attn,
                  n_samples, base_model, device):
    """
    Fast-DetectGPT zero-shot AI detector (Bao et al., 2023).

    \b
    References:
    ===========
        Bao, Guangsheng, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2023.
        “Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional
        Probability Curvature.” arXiv [Cs.CL]. arXiv. https://arxiv.org/abs/2310.05130.
    """
    from genai_detection.detectors.fastdetectgpt import FastDetectGPT

    detector = FastDetectGPT(
        base_model=base_model,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command(cls=DetectorCommand, with_model_path=True, with_gpu_opts=True)
@click.option('--device', help='Model device', default='auto')
def supervised_hf(model, input_dataset, output_csv, dataset_split, batch_size, quantize, flash_attn, device):
    """
    Generative AI detector using a fine-tuned sequence classification model.

    Requires a fine-tuned transformer sequence classification. Use the finetune train-seq-cls
    subcommand to train save suitable models.
    """
    from genai_detection.detectors.supervised import SupervisedDetector
    detector = SupervisedDetector(
        hf_model=model,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        device_map=device)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command(cls=DetectorCommand, with_model_path=True, with_gpu_opts=False)
def supervised_sklearn(model, input_dataset, output_csv, dataset_split, batch_size):
    """
    Generative AI detector using a trained scikit-learn model.

    Requires a trained scikit-learn classification model. Use the finetune train-tf-idf
    subcommand to train save suitable models.
    """
    from genai_detection.detectors.supervised import SupervisedDetector
    detector = SupervisedDetector(sklearn_model=model)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command(cls=DetectorCommand)
def ppmd(input_dataset, output_csv, dataset_split, batch_size):
    """
    Baseline AI detector using PPMd compression-based cosine.

    \b
    References:
    ===========
        Sculley, D., and C. E. Brodley. 2006. “Compression and Machine Learning: A New Perspective
        on Feature Space Vectors.” In Data Compression Conference (DCC’06), 332–41. IEEE.

        Halvani, Oren, Christian Winter, and Lukas Graner. 2017. “On the Usefulness of Compression
        Models for Authorship Verification.” In ACM International Conference Proceeding Series. Vol.
        Part F1305. Association for Computing Machinery. https://doi.org/10.1145/3098954.3104050.
    """

    from genai_detection.detectors.ppmd import PPMdDetector
    detector = PPMdDetector()
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command(cls=DetectorCommand)
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-r', '--rounds', type=int, default=35, show_default=True, help='Deconstruction rounds')
@click.option('-t', '--top-n', type=int, default=200, show_default=True, help='Number of top features')
@click.option('-c', '--cv-folds', type=int, default=10, show_default=True, help='Cross-validation folds')
@click.option('-d', '--n-delete', type=int, default=4, show_default=True,
              help='Features to eliminate per round')
@click.option('-s', '--chunk-size', type=int, default=700, show_default=True, help='Chunk sample size')
@click.option('-n', '--n-chunks', type=int, default=60, show_default=True, help='Number of chunks to sample')
def unmasking(input_dataset, output_csv, dataset_split, batch_size, rounds, top_n,
              cv_folds, n_delete, chunk_size, n_chunks):
    """
    Baseline AI detector using authorship unmasking.

    \b
    References:
    ===========
        Koppel, Moshe, and Jonathan Schler. 2004. “Authorship Verification as a One-Class
        Classification Problem.” In Proceedings, Twenty-First International Conference on
        Machine Learning, ICML 2004, 489–95.

        Bevendorff, Janek, Benno Stein, Matthias Hagen, and Martin Potthast. 2019. “Generalizing
        Unmasking for Short Texts.” In Proceedings of the 2019 Conference of the North, 654–59.
        Stroudsburg, PA, USA: Association for Computational Linguistics.
    """
    from genai_detection.detectors.unmasking import UnmaskingDetector
    detector = UnmaskingDetector(rounds, top_n, cv_folds, n_delete, chunk_size, n_chunks)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command(cls=DetectorCommand)
@click.option('-w', '--word-length', type=click.FloatRange(1.), help='Word length threshold', default=5.1)
@click.option('--shorter-is-ai', is_flag=True, help='Whether shorter lengths are AI')
def word_length(input_dataset, output_csv, dataset_split, batch_size, word_length, shorter_is_ai):
    """
    AI detection baseline using average word length.
    """
    from genai_detection.detectors.word_length import WordLengthDetector

    detector = WordLengthDetector(word_length=word_length, shorter_is_ai=shorter_is_ai)
    detect(detector, input_dataset, output_csv, dataset_split, batch_size)


@main.command()
@click.argument('input_csv', type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option('-o', '--output-csv', type=click.Path(file_okay=True, exists=False),
              help='Path for CSV output file with predictions', default='data/ensemble-predictions.csv')
@click.option('-w', '--weight', type=click.FloatRange(0.), help='Input weights', multiple=True)
def majority(input_csv, output_csv, weight):
    """
    Majority decision of previously run detectors.
    """

    if len(input_csv) < 2:
        raise click.UsageError('Need at least two input CSV files.')

    if not weight:
        weight = np.ones(len(input_csv))
    elif len(weight) != len(input_csv):
        raise click.UsageError('Number of weights must match number of inputs.')
    weight = np.array(weight) / np.sum(weight)

    df = pd.read_csv(input_csv[0], index_col='id')[['pred_label']]
    for i in range(1, len(input_csv)):
        df = df.join(pd.read_csv(input_csv[i], index_col='id')[['pred_label']], how='inner', rsuffix=f'_{i}')

    df = (df * weight).sum(axis=1).to_frame('score')
    df.insert(0, 'pred_label', (df['score'] > .5).astype(int))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=True)
    print(f'Predictions written to {output_csv}.')
