import csv
import json
from pathlib import Path
import os
import sys
import types
import typing as t
import uuid

import click
from datasets import ClassLabel, concatenate_datasets, Dataset, DatasetDict, Features, load_dataset, Value
from tqdm import tqdm


DATASET_FEATURES = Features({
    'id': Value('string'),
    'text': Value('string'),
    'label': ClassLabel(names=['human', 'machine']),
    'model': Value('string'),
})


@click.group()
def main():
    pass


def _txt_dir_to_dataset(basedir, is_human, recursive, model_name_parent):
    def _read_files(paths):
        for p in paths:  # type: Path
            if t := open(p, errors='ignore').read().strip():
                yield {
                    'id': '/'.join([p.parents[1].name, p.parents[0].name, p.stem]),
                    'text': t,
                    'label': 'human' if is_human else 'machine',
                    'model': p.parents[min(model_name_parent, len(p.parents) - 1)].name if not is_human else 'human'
                }

    if recursive:
        p = list(Path(basedir).rglob('**/*.txt'))
    else:
        p = list(Path(basedir).glob('*.txt'))

    if not p:
        raise click.UsageError(f'No text files found in "{basedir}"')
    return Dataset.from_generator(_read_files, features=DATASET_FEATURES, gen_kwargs={'paths': p})


def _load_dataset_and_adapt_schema(load_fn, filename, column_map, human_labels, drop_cols=None, **load_kwargs):
    human_labels = {l.lower() for l in human_labels}
    label_col = [k for k, v in column_map.items() if v == 'label']
    label_col = label_col[0] if label_col else 'label'
    model_col = [k for k, v in column_map.items() if v == 'model']
    model_col = model_col[0] if model_col else 'model'

    def _adapt_schema(r):
        d = {
            'id': str(r.get('id', uuid.uuid4())),
            'model': 'human' if str(r[model_col]).lower() in human_labels else r.get(model_col, 'machine')
        }
        if label_col in r:
            d.update({'label': (str(r[label_col]).lower() in human_labels) and 'human' or 'machine'})
        return d

    def _split_pairs(batch):
        keys_special = ['id', 'text', 'label', 'model', '__machine_text', '__human_text']
        keys = [k for k in batch if k not in keys_special]
        d_new = {k: [] for k in keys + keys_special}
        for i, (t_h, t_m) in enumerate(zip(batch.get('__human_text', batch.get('text')), batch['__machine_text'])):
            for t_, l_ in ((t_h, 'human'), (t_m, 'machine')):
                # Do not add null / empty texts, but keep the other text if available
                if t_ is None or not t_.strip():
                    continue
                # Repeat non-special keys
                for k in keys:
                    d_new[k].append(batch[k][i])
                # Fill in special keys
                d_new['id'].append(str(uuid.uuid4()))
                d_new['text'].append(t_)
                d_new['label'].append(l_)
                d_new['__human_text'].append('')
                d_new['__machine_text'].append('')
                if 'model' in batch:
                    d_new['model'].append('human' if l_ == 'human' else batch['model'][i])
        return d_new

    ds: Dataset = load_fn(filename, **load_kwargs)
    if drop_cols:
        ds = ds.remove_columns([c for c in drop_cols if c in ds.column_names])
    ds = ds.rename_columns({k: v for k, v in column_map.items()
                            if k in ds.column_names
                            and k != v and v not in ['label', 'model']
                            and v not in ds.column_names})
    ds = ds.map(_adapt_schema, num_proc=len(os.sched_getaffinity(0)) - 1)

    if '__machine_text' in ds.column_names or '__human_text' in ds.column_names:
        ds = ds.map(_split_pairs, num_proc=len(os.sched_getaffinity(0)) - 1, batched=True)
        ds = ds.remove_columns(['__human_text', '__machine_text'])

    # Make sure id is the first column
    f = Features({'id': ds.features['id'], **ds.features})
    f.update({k: v for k, v in DATASET_FEATURES.items() if k in f})
    return ds.cast(features=f)


def _load_jsonl(filename, field=None):
    objs = []
    with open(filename, errors='ignore') as f:
        for l in f:
            if not l.strip():
                continue
            j = json.loads(l)
            if field:
                j = j[field]
            for k in j:
                if type(j[k]) not in [str, int, float, types.NoneType]:
                    j[k] = json.dumps(j[k])
            objs.append(j)
    return Dataset.from_list(objs)


def _load_json_single(filename):
    obj = json.load(open(filename, 'r', errors='ignore'))
    if type(obj) is dict:
        obj = list(obj.values())
    for i, record in enumerate(obj):
        for k, v in list(record.items()):
            if type(v) is dict:
                for k_, v_ in v.items():
                    obj[i][f'{k}_{k_}'] = json.dumps(v_) if type(v_) in [list, dict] else str(v_)
            else:
                obj[i][k] = json.dumps(v) if type(v) is list else str(v)
    return Dataset.from_list(obj)


def create_dataset_split(ds_train, test_ids=None, test_split_size=None, shuffle=False, seed=None) \
        -> t.Tuple[Dataset, t.Optional[Dataset]]:
    """
    Create a class-stratified train/test split of a given Dataset.

    If no test IDs are given and ``test_split_size`` is ``None``, the input dataset will be returned
    unchanged or shuffled (if ``shuffle`` is ``True``).

    :param ds_train: input dataset
    :param test_ids: list with IDs to build the test set from
    :param test_split_size: number of ratio of test examples (if ``test_ids`` is not given)
    :param shuffle: shuffle the dataset
    :param seed: random seed
    :return: dataset split
    """
    def _is_in_id_list(r, id_list):
        for i in id_list:
            if r['id'].endswith(i):
                return True
        return False

    ds_test: t.Optional[Dataset] = None
    if test_ids is not None:
        test_ids = set(l.strip() for l in test_ids.readlines())
        ds_test = ds_train.filter(lambda r: _is_in_id_list(r, test_ids))
        ds_train = ds_train.filter(lambda r: not _is_in_id_list(r, test_ids))
    elif test_split_size:
        ds_train, ds_test = ds_train.train_test_split(test_size=test_split_size,
                                                      stratify_by_column='label' if shuffle else None,
                                                      shuffle=shuffle,
                                                      seed=seed).values()
    elif shuffle:
        ds_train = ds_train.shuffle(seed=seed)

    return ds_train, ds_test


# noinspection PyProtectedMember,PyClassVar
class IntOrFloatRange(click.types._NumberRangeBase):
    name = 'integer or float range'

    def __init__(self, min_int=None, max_int=None, min_float=None, max_float=None,
                 min_open=False, max_open=False, clamp=False):
        self.min_int = min_int
        self.max_int = max_int
        self.min_float = min_float
        self.max_float = max_float
        super().__init__(min_open=min_open, max_open=max_open, clamp=clamp)

    def convert(self, value, param, ctx):
        if '.' in value:
            self.min = self.min_float
            self.max = self.max_float
            self._number_class = float
        else:
            self.min = self.min_int
            self.max = self.max_int
            self._number_class = int

        return super().convert(value, param, ctx)

    def _describe_range(self) -> str:
        self.min = self.min_int
        self.max = self.max_int
        r_int = super()._describe_range()

        self.min = self.min_float
        self.max = self.max_float
        r_float = super()._describe_range()

        return f'{r_int}, {r_float}'


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_file', type=click.File('w'))
@click.option('-m', '--keep-model', help='Keep only these models', multiple=True,
              default=['human', 'chatgpt', 'llama-chat', 'mistral', 'mistral-chat', 'gpt4', 'gpt3'])
@click.option('-l', '--min-length', type=click.IntRange(0), help='Minimum text length in characters',
              default=750)
@click.option('-a', '--attacks', is_flag=True, help='Include adversarial attacks')
def filter_raid(input_file, output_file, keep_model, min_length, attacks):
    """
    Filter RAID dataset (https://raid-bench.xyz/).

    Creates subsets of the original dataset with certain LLMs and short texts removed.
    """
    csv.field_size_limit(sys.maxsize)
    reader = csv.DictReader(input_file, delimiter=',')
    writer = csv.DictWriter(output_file, reader.fieldnames, delimiter=',')
    writer.writeheader()
    num_in = 0
    num_out = 0
    for r in tqdm(reader, total=reader.line_num, desc='Filtering RAID dataset'):
        num_in += 1
        if (len(r['generation']) <= min_length
                or r['model'] not in keep_model
                or (not attacks and r['attack'] != 'none')):
            continue
        writer.writerow(r)
        num_out += 1
    input_file.close()
    output_file.close()
    print(f'Input records: {num_in + 1:,}, Output records: {num_out:,}')


@main.command()
@click.option('-h', '--human-dir', type=click.Path(exists=True), multiple=True,
              help='Directory with human texts')
@click.option('-m', '--machine-dir', type=click.Path(exists=True), multiple=True,
              help='Directory with machine texts')
@click.option('-r', '--recursive', is_flag=True, help='Scan text directories recursively')
@click.option('--model-name-parent', type=click.IntRange(0), default=0,
              help='Which direct parent dir name corresponds to model name')
@click.option('-c', '--csv-file', type=click.Path(exists=True,), multiple=True,
              help='CSV file with human and machine texts')
@click.option('-j', '--json-file', type=click.Path(exists=True), multiple=True,
              help='Line-delimited JSON file with human and machine texts')
@click.option('--json-file-single', type=click.Path(exists=True), multiple=True,
              help='JSON file containing a single list or dict')
@click.option('-o', '--output', type=click.Path(dir_okay=False, exists=False, writable=True),
              default='data/dataset-converted')
@click.option('-v', '--val-split-size', type=IntOrFloatRange(0, None, 0., 1.),
              help='Size of validation split (relative or absolute)')
@click.option('-t', '--test-split-size', type=IntOrFloatRange(0, None, 0., 1.),
              help='Size of test split (relative or absolute, if --test-ids not given)')
@click.option('--val-ids', type=click.File('r'), help='Text file with text ID suffixes for validation split')
@click.option('--test-ids', type=click.File('r'), help='Text file with text ID suffixes for test split')
@click.option('--no-shuffle', is_flag=True, help='Do not shuffle dataset')
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
@click.option('--id-col', default='id', help='ID column in CSV or JSON')
@click.option('--text-col', default='text', help='Text column in CSV or JSON')
@click.option('--human-text-col', multiple=True,
              help='Additional human text column in CSV or JSON if rows are text pairs')
@click.option('--machine-text-col', multiple=True,
              help='Additional machine text column in CSV or JSON if rows are text pairs')
@click.option('--label-col', default='label', help='Label column in CSV or JSON')
@click.option('--model-col', default='model', help='Model name column in CSV or JSON')
@click.option('--drop-col', help='CSV or JSON columns to drop', multiple=True)
@click.option('--human-label', help='CSV label value for human class', multiple=True)
@click.option('--dataset-field', help='JSON field containing the dataset')
def convert(human_dir, machine_dir, recursive, model_name_parent, csv_file, json_file, json_file_single,
            output, val_split_size, test_split_size, val_ids, test_ids, no_shuffle, seed,
            id_col, text_col, human_text_col, machine_text_col, label_col, model_col, drop_col, human_label,
            dataset_field):
    """
    Create a Huggingface dataset from folders with text files or CSV / JSON files.

    Text file folders are searched recursively for ``*.txt`` files and the resulting dataset rows will use
    the names of the two immediate parent directories and the filename (without ``.txt`` extension) as record IDs.

    CSV and JSON files need to have at least a ``text`` and a ``label`` column (IDs will be autogenerated if not given).

    With ``--val-ids`` / ``--test-ids``, you can supply a text file with text ID suffixes (excluding ``.txt``
    extension). Any input files matching this suffix will be part only of the validation / test split.
    Unlike ``--val-split-size`` / ``--test-split-size``, this allows you to avoid domain leaks between training and
    test (e.g. if ``human/x/art-001.txt`` and ``llm1/x/art-001.txt`` are from the same domain and should not be
    split between training and test).
    """

    if not human_dir and not machine_dir and not csv_file and not json_file and not json_file_single:
        raise click.UsageError('Need to specify at least one input file or directory.')

    human_dir = [d for d in human_dir if os.path.isdir(d)]
    machine_dir = [d for d in machine_dir if os.path.isdir(d)]
    csv_file = [d for d in csv_file if os.path.isfile(d)]
    json_file = [d for d in json_file if os.path.isfile(d)]
    json_file_single = [d for d in json_file_single if os.path.isfile(d)]
    human_label = human_label or ['0', 'human']

    if type(val_split_size) is float and type(test_split_size) is float and \
            val_split_size + test_split_size > 1.0 and test_ids is None and val_ids is None:
        raise click.UsageError('Sum of validation and test split sizes must be <= 1.0!')

    column_map = {
        id_col: 'id',
        text_col: 'text',
        label_col: 'label',
        model_col: 'model',
    }
    if human_text_col:
        column_map.update({h: '__human_text' for h in human_text_col})
    if machine_text_col:
        column_map.update({m: '__machine_text' for m in machine_text_col})

    ds_train = [
        *[_txt_dir_to_dataset(d, True, recursive, model_name_parent) for d in human_dir],
        *[_txt_dir_to_dataset(d, False, recursive, model_name_parent) for d in machine_dir],
        *[_load_dataset_and_adapt_schema(
            Dataset.from_csv, c, column_map, human_label, drop_col) for c in csv_file],
        *[_load_dataset_and_adapt_schema(
            _load_jsonl, c, column_map, human_label, drop_col, field=dataset_field) for c in json_file],
        *[_load_dataset_and_adapt_schema(
            _load_json_single, c, column_map, human_label, drop_col) for c in json_file_single],
    ]
    if not ds_train:
        raise click.UsageError('Empty dataset!')

    ds_train = concatenate_datasets(ds_train)
    if 'text' not in ds_train.column_names:
        raise click.UsageError('Dataset has no text column! Specify a '
                               'column mapping if the column is named differently!')
    if not no_shuffle:
        ds_train = ds_train.shuffle(seed=seed)

    if type(test_split_size) is float and test_split_size > 0:
        test_split_size = int(len(ds_train) * test_split_size)
    if type(val_split_size) is float and val_split_size > 0:
        val_split_size = int(len(ds_train) * val_split_size)

    ds_train, ds_test = create_dataset_split(ds_train,
                                             test_ids=test_ids,
                                             test_split_size=test_split_size,
                                             shuffle=not no_shuffle,
                                             seed=seed)
    ds_train, ds_val = create_dataset_split(ds_train,
                                            test_ids=val_ids,
                                            test_split_size=val_split_size,
                                            shuffle=not no_shuffle,
                                            seed=seed)

    d = DatasetDict({'train': ds_train})
    if ds_val:
        d.update({'validation': ds_val})
    if ds_test:
        d.update({'test': ds_test})
    d.save_to_disk(output)


@main.command()
@click.argument('hf_dataset')
@click.option('-o', '--output', type=click.Path(dir_okay=False, exists=False, writable=True),
              default='data/dataset-converted')
@click.option('-v', '--val-split-size', type=IntOrFloatRange(0, None, 0., 1.),
              help='Size of validation split if does not exist (relative or absolute)')
@click.option('-t', '--test-split-size', type=IntOrFloatRange(0, None, 0., 1.),
              help='Size of test split if does not exist (relative or absolute, if --test-ids not given)')
@click.option('--no-shuffle', is_flag=True, help='Do not shuffle when creating new splits')
@click.option('--id-col', default='id', help='ID column')
@click.option('--text-col', default='text', help='Text column')
@click.option('--label-col', default='label', help='Label column')
@click.option('--model-col', default='model', help='Model name column')
@click.option('--human-label', help='Label value for human class', multiple=True)
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
def adapt_hf_dataset(hf_dataset, output, val_split_size, test_split_size, no_shuffle,
                     id_col, text_col, label_col, model_col, human_label, seed):
    """
    Adapt schema of an existing Huggingface dataset.
    """

    ds_dict = load_dataset(hf_dataset)
    ds_train = ds_dict['train']
    ds_val = None
    ds_test = None
    human_label = human_label or ['0', 'human']

    if type(val_split_size) is float and type(test_split_size) is float and \
            val_split_size + test_split_size > 1.0:
        raise click.UsageError('Sum of validation and test split sizes must be <= 1.0!')

    if 'test' in ds_dict:
        ds_test = ds_dict['test']
    elif type(test_split_size) is float and test_split_size > 0:
        test_split_size = int(len(ds_train) * test_split_size)
        ds_train, ds_test = create_dataset_split(ds_train,
                                                 test_split_size=test_split_size,
                                                 shuffle=not no_shuffle,
                                                 seed=seed)

    if 'validation' in ds_dict:
        ds_val = ds_dict['validation']
    elif type(val_split_size) is float and val_split_size > 0:
        val_split_size = int(len(ds_train) * val_split_size)
        ds_train, ds_val = create_dataset_split(ds_train,
                                                test_split_size=val_split_size,
                                                shuffle=not no_shuffle,
                                                seed=seed)

    d = DatasetDict()
    ds_train = _load_dataset_and_adapt_schema(lambda _: ds_train, '', {
        id_col: 'id',
        text_col: 'text',
        label_col: 'label',
        model_col: 'model',
    }, human_label)
    d.update({'train': ds_train})

    if ds_val:
        ds_val = _load_dataset_and_adapt_schema(lambda _: ds_val, '', {
            id_col: 'id',
            text_col: 'text',
            label_col: 'label',
            model_col: 'model',
        }, human_label)
        d.update({'validation': ds_val})

    if ds_test:
        ds_test = _load_dataset_and_adapt_schema(lambda _: ds_test, '', {
            id_col: 'id',
            text_col: 'text',
            label_col: 'label',
            model_col: 'model',
        }, human_label)
        d.update({'test': ds_test})

    d.save_to_disk(output)
