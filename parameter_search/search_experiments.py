from automusicgen.util.constants import SongPartitionMethod, Songs
from .testing_parameters import TransformerParameters
from automusicgen.data.tokenize.midi_tokenizer import VOCAB_SIZE

GRAVITY_FALLS_EXPERIMENT = [
        TransformerParameters(model_name='GravityFalls5', epochs=5, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.RandomSubstrings25000),
        TransformerParameters(model_name='GravityFalls10', epochs=10, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.RandomSubstrings25000),
        # TransformerParameters(model_name='GravityFalls20', epochs=20, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.RandomSubstrings25000),
        # TransformerParameters(model_name='GravityFalls35', epochs=35, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.RandomSubstrings1000),
        # TransformerParameters(model_name='GravityFalls50', epochs=50, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.RandomSubstrings1000),
]

GRAVITY_FALLS_ALL_PARTITIONS_EXPERIMENT = [
        TransformerParameters(model_name='GravityFallsAllPartitions1', epochs=1, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.AllSubstrings),
        # TransformerParameters(model_name='GravityFallsAllPartitions3', epochs=3, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.AllSubstrings),
        # TransformerParameters(model_name='GravityFallsAllPartitions5', epochs=5, input_dict_size=VOCAB_SIZE, training_songs=[Songs.GRAVITY_FALLS], test_song=Songs.GRAVITY_FALLS, partition_method=SongPartitionMethod.AllSubstrings),
]

UNDER_THE_SEA_EXPERIMENT = [
        TransformerParameters(model_name='UnderTheSea20', epochs=20, input_dict_size=VOCAB_SIZE, training_songs=[Songs.UNDER_THE_SEA], test_song=Songs.UNDER_THE_SEA, partition_method=SongPartitionMethod.AllSubstringsLength70),        
        TransformerParameters(model_name='UnderTheSea50', epochs=50, input_dict_size=VOCAB_SIZE, training_songs=[Songs.UNDER_THE_SEA], test_song=Songs.UNDER_THE_SEA, partition_method=SongPartitionMethod.AllSubstringsLength70),        
        # TransformerParameters(model_name='UnderTheSea10', epochs=10, input_dict_size=VOCAB_SIZE, training_songs=[Songs.UNDER_THE_SEA], test_song=Songs.UNDER_THE_SEA, partition_method=SongPartitionMethod.RandomSubstrings10000),        
]

SWEET_CHILD_O_MINE_EXPERIMENT = [
        TransformerParameters(model_name='SweetChildOMine20', epochs=20, input_dict_size=VOCAB_SIZE, training_songs=[Songs.SWEET_CHILD_O_MINE], test_song=Songs.SWEET_CHILD_O_MINE, partition_method=SongPartitionMethod.AllSubstringsLength70),        
        # TransformerParameters(model_name='SweetChildOMine3', epochs=3, input_dict_size=VOCAB_SIZE, training_songs=[Songs.SWEET_CHILD_O_MINE], test_song=Songs.SWEET_CHILD_O_MINE, partition_method=SongPartitionMethod.RandomSubstrings1000),        
        TransformerParameters(model_name='SweetChildOMine50', epochs=50, input_dict_size=VOCAB_SIZE, training_songs=[Songs.SWEET_CHILD_O_MINE], test_song=Songs.SWEET_CHILD_O_MINE, partition_method=SongPartitionMethod.AllSubstringsLength70),        
]
TETRIS_EXPERIMENT = [
        TransformerParameters(model_name='Tetris2', epochs=2, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TETRIS], test_song=Songs.TETRIS, partition_method=SongPartitionMethod.RandomSubstrings1000),        
        TransformerParameters(model_name='Tetris5', epochs=5, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TETRIS], test_song=Songs.TETRIS, partition_method=SongPartitionMethod.RandomSubstrings1000),        
]

TETRIS_FULL_EXPERIMENT = [
        TransformerParameters(model_name='Tetris1', epochs=1, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TETRIS], test_song=Songs.TETRIS, partition_method=SongPartitionMethod.AllSubstrings),        
        TransformerParameters(model_name='Tetris2', epochs=2, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TETRIS], test_song=Songs.TETRIS, partition_method=SongPartitionMethod.AllSubstrings),        
]

CLASSICAL_EXPERIMENT = [
        TransformerParameters(model_name='Classical5', epochs=5, input_dict_size=VOCAB_SIZE, training_songs=[Songs.CHOPIN, Songs.TCHAIKOVSKY], test_song=Songs.CHOPIN, partition_method=SongPartitionMethod.AllSubstringsLength50),
        TransformerParameters(model_name='Classical20', epochs=20, input_dict_size=VOCAB_SIZE, training_songs=[Songs.CHOPIN, Songs.TCHAIKOVSKY], test_song=Songs.CHOPIN, partition_method=SongPartitionMethod.AllSubstringsLength50),
]

CHOPIN_EXPERIMENT = [
        TransformerParameters(model_name='Chopin50', epochs=50, input_dict_size=VOCAB_SIZE, training_songs=[Songs.CHOPIN], test_song=Songs.CHOPIN, partition_method=SongPartitionMethod.AllSubstringsLength70),
        TransformerParameters(model_name='Chopin85', epochs=85, input_dict_size=VOCAB_SIZE, training_songs=[Songs.CHOPIN], test_song=Songs.CHOPIN, partition_method=SongPartitionMethod.AllSubstringsLength70),
]

BACH_EXPERIMENT = [
        TransformerParameters(model_name='Bach50', epochs=50, input_dict_size=VOCAB_SIZE, training_songs=[Songs.BACH], test_song=Songs.BACH, partition_method=SongPartitionMethod.AllSubstringsLength70),
        TransformerParameters(model_name='Bach85', epochs=85, input_dict_size=VOCAB_SIZE, training_songs=[Songs.BACH], test_song=Songs.BACH, partition_method=SongPartitionMethod.AllSubstringsLength70),
]

TCHAIKOVSKY_EXPERIMENT = [
        TransformerParameters(model_name='Tchaikovsky50', epochs=50, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TCHAIKOVSKY], test_song=Songs.TCHAIKOVSKY, partition_method=SongPartitionMethod.AllSubstringsLength70),
        TransformerParameters(model_name='Tchaikovsky85', epochs=85, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TCHAIKOVSKY], test_song=Songs.TCHAIKOVSKY, partition_method=SongPartitionMethod.AllSubstringsLength70),
]

EXPERIMENT_OF_DOOM = [
        TransformerParameters(model_name='Doom10', epochs=10, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TCHAIKOVSKY, Songs.MARIO_MEDLEY, Songs.MOONLIGHT_SONATA, Songs.CHOPIN], test_song=Songs.TCHAIKOVSKY, partition_method=SongPartitionMethod.AllSubstringsLength70),
        TransformerParameters(model_name='Doom25', epochs=25, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TCHAIKOVSKY, Songs.MARIO_MEDLEY, Songs.MOONLIGHT_SONATA, Songs.CHOPIN], test_song=Songs.TCHAIKOVSKY, partition_method=SongPartitionMethod.AllSubstringsLength70),
        TransformerParameters(model_name='Doom50', epochs=50, input_dict_size=VOCAB_SIZE, training_songs=[Songs.TCHAIKOVSKY, Songs.MARIO_MEDLEY, Songs.MOONLIGHT_SONATA, Songs.CHOPIN], test_song=Songs.TCHAIKOVSKY, partition_method=SongPartitionMethod.AllSubstringsLength70),
]

MARIO_EXPERIMENT = [
        TransformerParameters(model_name='Mario10', epochs=10, input_dict_size=VOCAB_SIZE, training_songs=[Songs.MARIO_MEDLEY], test_song=Songs.MARIO_MEDLEY, partition_method=SongPartitionMethod.AllSubstringsLength70),
        TransformerParameters(model_name='Mario25', epochs=25, input_dict_size=VOCAB_SIZE, training_songs=[Songs.MARIO_MEDLEY], test_song=Songs.MARIO_MEDLEY, partition_method=SongPartitionMethod.AllSubstringsLength70),

]


# TEST_EXPERIMENT = [
#     TransformerParameters(model_name='TestModel3', epochs=3, training_songs=[Songs.TEST_SONG], test_song=Songs.TEST_SONG),
#     TransformerParameters(model_name='TestModel5', epochs=5, training_songs=[Songs.TEST_SONG], test_song=Songs.TEST_SONG),
# ]

# TWINKLE_TWINKLE_EXPERIMENT = [
#     TransformerParameters(model_name='Twinkle5', epochs=5, training_songs=[Songs.TWINKLE_TWINKLE], test_song=Songs.TWINKLE_TWINKLE),
#     TransformerParameters(model_name='Twinkle10', epochs=10, training_songs=[Songs.TWINKLE_TWINKLE], test_song=Songs.TWINKLE_TWINKLE),
#     TransformerParameters(model_name='Twinkle20', epochs=20, training_songs=[Songs.TWINKLE_TWINKLE], test_song=Songs.TWINKLE_TWINKLE),
#     TransformerParameters(model_name='Twinkle35', epochs=35, training_songs=[Songs.TWINKLE_TWINKLE], test_song=Songs.TWINKLE_TWINKLE),
#     TransformerParameters(model_name='Twinkle50', epochs=50, training_songs=[Songs.TWINKLE_TWINKLE], test_song=Songs.TWINKLE_TWINKLE),
# ]

# ODE_TO_JOY_EXPERIMENT = [
#     TransformerParameters(model_name='OdeToJoy5', epochs=5, training_songs=[Songs.ODE_TO_JOY], test_song=Songs.ODE_TO_JOY),
#     TransformerParameters(model_name='OdeToJoy10', epochs=10, training_songs=[Songs.ODE_TO_JOY], test_song=Songs.ODE_TO_JOY),
#     TransformerParameters(model_name='OdeToJoy20', epochs=20, training_songs=[Songs.ODE_TO_JOY], test_song=Songs.ODE_TO_JOY),
#     TransformerParameters(model_name='OdeToJoy35', epochs=35, training_songs=[Songs.ODE_TO_JOY], test_song=Songs.ODE_TO_JOY),
#     TransformerParameters(model_name='OdeToJoy50', epochs=50, training_songs=[Songs.ODE_TO_JOY], test_song=Songs.ODE_TO_JOY),
# ]

# CHAOS_EXPERIMENT = [
#     TransformerParameters(model_name='Chaos5', epochs=5, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE),
#     TransformerParameters(model_name='Chaos10', epochs=10, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE),
#     TransformerParameters(model_name='Chaos20', epochs=20, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE),
#     TransformerParameters(model_name='Chaos35', epochs=35, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE),
# ]

# JACK_EXPERIMENT = [
#     TransformerParameters(model_name='Jack5', epochs=5, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE, partition_method=SongPartitionMethod.RandomSubstrings500),
#     TransformerParameters(model_name='Jack10', epochs=10, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE, partition_method=SongPartitionMethod.RandomSubstrings500),
#     TransformerParameters(model_name='Jack20', epochs=20, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE, partition_method=SongPartitionMethod.RandomSubstrings500),
#     TransformerParameters(model_name='Jack35', epochs=35, training_songs=[Songs.ODE_TO_JOY, Songs.TWINKLE_TWINKLE, Songs.OLD_MCDONALD], test_song=Songs.TWINKLE_TWINKLE, partition_method=SongPartitionMethod.RandomSubstrings500),
# ]
