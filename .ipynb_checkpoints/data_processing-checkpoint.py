import os
import glob
import torch
import numpy as np
import librosa as lr
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.files = glob.glob(os.path.join(file_dir, '*.mp3')) # currently just mp3
    
    def __len__(self):
        # length of dataset
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        y, _ = lr.load(file)
        y = self.mu_law_encode(y)
        y = torch.from_numpy(y.reshape(-1)).type(torch.LongTensor) #convert to tensor
        return file, y
        
    def mu_law_encode(self, audio, quantization_channels=256):
        #Quantizes waveform amplitudes 
        safe_audio = np.minimum(np.abs(audio), 1.0)
        signal = np.sign(audio) * np.log(1 + safe_audio * quantization_channels) / np.log(1 + quantization_channels)
        
        return ((signal + 1) / 2 * quantization_channels + 0.5).astype(int) #discretize to 0~255