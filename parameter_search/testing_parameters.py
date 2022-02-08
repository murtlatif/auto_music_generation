
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.util.constants import Songs


class TestingParameters:
    model_name: str

    # Model parameters
    input_dict_size: int
    output_dict_size: int
    hidden_dim: int
    feedforward_hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float

    # Training parameters
    epochs: int
    learning_rate: float
    training_songs: list[str]

    # Evaluation parameters
    test_song: str

    def __init__(
        self, 
        model_name: str = 'TestModel',
        input_dict_size: int = len(MusicToken),
        output_dict_size: int = None,
        hidden_dim: int = 512,
        feedforward_hidden_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        epochs: int = 25,
        learning_rate: float = 1e-5,
        test_song: str = Songs.TWINKLE_TWINKLE,
        training_songs: str = [Songs.TWINKLE_TWINKLE],
    ):
        self.model_name = model_name

        self.input_dict_size = input_dict_size
        self.output_dict_size = output_dict_size
        self.hidden_dim = hidden_dim
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.training_songs = training_songs

        self.test_song = test_song

        self.parameter_dict = {
            'model_name': self.model_name,
            'input_dict_size': self.input_dict_size,
            'output_dict_size': self.output_dict_size,
            'hidden_dim': self.hidden_dim,
            'feedforward_hidden_dim': self.feedforward_hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'test_song': self.test_song,
            'training_songs': self.training_songs,
        }

    def __repr__(self) -> str:
        return f'Testing Parameters: {self.parameter_dict}'
