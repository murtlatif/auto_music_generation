
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.util.constants import SaveMode, SongPartitionMethod, Songs


class TransformerParameters:
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
    print_status: bool
    save_mode: SaveMode
    save_on_accuracy: bool
    partition_method: SongPartitionMethod

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
        training_songs: str = [Songs.GRAVITY_FALLS],
        print_status: bool = True,
        save_mode: SaveMode = SaveMode.SaveBest,
        save_on_accuracy: bool = True,
        partition_method: SongPartitionMethod = SongPartitionMethod.AllSubstrings,
        
        test_song: str = Songs.GRAVITY_FALLS,
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
        self.print_status = print_status
        self.save_mode = save_mode
        self.save_on_accuracy = save_on_accuracy
        self.partition_method = partition_method

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
            'training_songs': self.training_songs,
            'print_status': self.print_status,
            'save_mode': self.save_mode,
            'save_on_accuracy': self.save_on_accuracy,
            'partition_method': self.partition_method,

            'test_song': self.test_song,
        }

    def __repr__(self) -> str:
        return f'Transformer Parameters: {self.parameter_dict}'
