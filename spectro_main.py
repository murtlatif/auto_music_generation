import librosa
import numpy as np

from config import Config
from old.preprocess_music import get_dataset, get_label, spectrogram
from util.song_file import get_and_verify_song_path_from_config


def spectro_main():
    song_path = get_and_verify_song_path_from_config()
    model_state_path = Config.args.load_model_path

    chunk_size_s = 0.1
    overlap = 0
    waveform, sr = librosa.load(song_path)
    spectro = spectrogram(waveform, sr, chunk_size_s=chunk_size_s)
    label = get_label(spectro, model_state_path)

    # dataset, labels = get_dataset(
    #     song_path, model_state_path, chunk_size_s, overlap)

    # notes = [note_map[label] for label in np.argmax(labels, axis=1)]
    print(label)


if __name__ == '__main__':
    pass
