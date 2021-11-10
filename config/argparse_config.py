from argparse import ArgumentParser, Namespace


class ArgparseConfig:
    """Configuration for command line arguments"""
    _args: Namespace = None
    _parser: ArgumentParser = None

    def __contains__(self, key: str) -> bool:
        return key in self.args

    def __getitem__(self, key: str):
        if key not in self.args:
            raise KeyError(f'ArgparseConfig has no item \'{key}\'')
        return getattr(self.args, key)

    def __getattr__(self, name: str):
        return getattr(self.args, name)

    def __repr__(self) -> str:
        return str(vars(self.args))

    @property
    def parser(self):
        if self._parser is None:
            self._parser = ArgumentParser()
            self._add_arguments_to_parser(self._parser)
        return self._parser

    @property
    def args(self):
        if self._args is None:
            self._args = self.parser.parse_args()
            self._validate_args()
        return self._args

    def _validate_args(self):
        # If training the model, the number of epochs must be positive
        if self.args.train:
            assert self.args.epochs > 0, "Must train on at least 1 epoch"
            assert self.args.batch_size > 0, "Must have a batch_size value > 0"

    def _add_arguments_to_parser(self, parser: ArgumentParser):
        parser.add_argument(
            '-l', '--load-model', dest='load_model_path', help='File path to a .pt model')
        parser.add_argument('--song', help='Song file to use')
        parser.add_argument('--name', type=str, help='Name of model; used when saving model data')
        parser.add_argument('-t', '--train', action='store_true', help='Use this flag to train the model')
        parser.add_argument('-e', '--epochs', type=int, default=10, help='The number of epochs to train the model')
        parser.add_argument('--seed', type=int, default=29, help='The random seed to use')
        parser.add_argument('-i', '--interactive', action='store_true',
                            help='Interactive mode; will prompt for evaluation inputs')
        parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for training')
