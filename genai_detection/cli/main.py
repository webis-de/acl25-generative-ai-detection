import click
import importlib


class TransparentLazyGroup(click.Group):
    def __init__(self, *args, lazy_group, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_group = lazy_group

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        lazy = sorted(self._lazy_load().list_commands(ctx))
        return base + lazy

    def get_command(self, ctx, cmd_name):
        return self._lazy_load().get_command(ctx, cmd_name)

    def _lazy_load(self):
        mod, cmd = self.lazy_group.rsplit(":", 1)
        mod = importlib.import_module(mod)
        return getattr(mod, cmd)


@click.group(context_settings={'show_default': True})
def main():
    """Generative AI detection tools."""


@main.group(cls=TransparentLazyGroup, lazy_group='genai_detection.cli.detect:main')
def detect():
    """
    A collection of generative AI detection models.

    The input data for each model can be a Huggingface Dataset, a single CSV file, a single *.txt file,
    or a directory with *.txt files. If the input is a Huggingface dataset with multiple splits, you can
    select one with --dataset-split. If the input is a CSV, it is expected to have either one, two, or
    three columns with texts, texts and labels, or text IDs, texts, and labels, respectively.

    If the input does not adhere to any of these formats, please use the dataset subcommand for converting
    your dataset into a suitable format first.
    """


@main.group(cls=TransparentLazyGroup, lazy_group='genai_detection.cli.finetune:main')
def finetune():
    """Finetune pre-trained models for generative AI detection."""


@main.group(cls=TransparentLazyGroup, lazy_group='genai_detection.cli.dataset:main')
def dataset():
    """Generative AI detection dataset tools."""


if __name__ == '__main__':
    main()
