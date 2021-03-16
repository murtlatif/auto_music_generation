# Split input waveform into chunks of set second size
# Get label of note for the chunk using classifier
# Return labels and chunks

import torch
import torchaudio
import librosa
import numpy as np
from classifier_model import Net
from sklearn.preprocessing import OneHotEncoder

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
oneh_encoder = OneHotEncoder()

def spectrogram(waveform, sr, chunk_size_s=None, overlap=0, *, n_mels=80, n_fft=256):
    if chunk_size_s == None:
        return torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft)(torch.Tensor(waveform).reshape(1,-1))

    chunk_size = int(chunk_size_s * sr)
    # chunks = torch.FloatTensor(list(chunk_waveform(waveform, chunk_size, overlap)))
    chunks = torch.FloatTensor(list(autochunk_waveform(waveform, sr, chunk_size)))
    specgram = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft)(chunks)
    return specgram

def get_label(spec, net_state_path, *, n_mels=80, n_fft=256):
    # classify entire waveform first then assign all in chunks that label
    spec_data = torch.zeros(1, n_mels, n_fft)
    spec_data[:, :, :spec.shape[-1]] = spec  # PADDING WITH ZEROS
    net = Net()
    net.load_state_dict(torch.load(net_state_path))
    net.eval()
    output = net(spec_data)
    label = output.argmax(axis=1)
    return label

# Chunk waveform using a set chunk_size
def chunk_waveform(waveform, chunk_size, overlap=0):
    idx = 0
    # overlap_chunk = 0
    overlap_chunk = int(overlap * chunk_size)
    while idx + chunk_size - overlap_chunk <= len(waveform):
        chunk = waveform[idx:idx + chunk_size - overlap_chunk]
        temp = np.zeros(chunk_size)
        temp[:len(chunk)] = chunk
        yield temp
        idx += chunk_size - overlap_chunk
    chunk = waveform[idx:idx + chunk_size - overlap_chunk]
    if len(chunk) > chunk_size//2:
        temp = np.zeros(chunk_size)
        temp[:len(chunk)] = chunk
        yield temp

# Chunk waveform by automatically detecting notes
def autochunk_waveform(waveform, sr, chunk_size):
    onsets = librosa.onset.onset_detect(y=waveform, sr=sr, units='samples')
    prev_onset = 0
    for onset in onsets:
        chunk = waveform[prev_onset:onset]
        if len(chunk) < chunk_size:
            temp = np.zeros(chunk_size)
            temp[:len(chunk)] = chunk
            yield temp
        elif len(chunk) > chunk_size:
            chunks = list(chunk_waveform(chunk, chunk_size))
            for chunk in chunks:
                yield chunk
        else:
            yield chunk
        prev_onset = onset

def get_dataset(file_path, net_state_path, chunk_size_s, overlap, n_mels=80, n_fft=256):
    # chunk waveform and assign next chunk class as label
    waveform, sr = librosa.load(file_path)
    specs = spectrogram(waveform, sr, chunk_size_s=chunk_size_s, overlap=overlap, n_mels=n_mels, n_fft=n_fft)
    spec_data = torch.zeros(specs.shape[0], n_mels, n_fft)
    spec_data[:, :, :specs.shape[-1]] = specs  # PADDING WITH ZEROS

    labels = []
    for spec in specs:
        label = get_label(spec, net_state_path=net_state_path, n_mels=n_mels, n_fft=n_fft)
        labels.append(label.item())
    labels = np.asarray(labels)
    labels = oneh_encoder.fit_transform(labels.reshape(-1, 1)).toarray()
    return spec_data[:-1], labels[1:]