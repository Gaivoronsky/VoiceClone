from typing import List

import torch
from torch.utils.data import Dataset
from pandas import DataFrame

from utils.audio import Audio


class VCTKDataset(Dataset):
    def __init__(self, df: DataFrame, hp, train: bool = True):
        self.df = df
        self.hp = hp
        self.train = train
        self.data = self._generate_data(self.df)
        self.audio = Audio(hp)

    def _generate_data(self, df: DataFrame) -> List[tuple]:
        data = []
        old_idx = df.index[0]
        for idx in df.index[1:]:
            if df.loc[idx].utterance_id == df.loc[old_idx].utterance_id and \
                    df.loc[idx].speaker_id != df.loc[old_idx].speaker_id:
                data.append((df.loc[old_idx].wav, df.loc[idx].wav))
            old_idx = idx
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target_wav = sample[0][0, : int(self.hp.data.audio_len * self.hp.audio.sample_rate)].numpy()
        raw_wav = sample[1][0, : int(self.hp.data.audio_len * self.hp.audio.sample_rate)].numpy()

        dvec_mel = self.audio.get_mel(target_wav)
        dvec_mel = torch.from_numpy(dvec_mel)
        if self.train:
            target_mag, _ = self.audio.wav2spec(target_wav)
            raw_mag, _ = self.audio.wav2spec(raw_wav)
            target_mag = torch.from_numpy(target_mag)
            raw_mag = torch.from_numpy(raw_mag)
            return dvec_mel, target_mag, raw_mag
        else:
            target_mag, _ = self.audio.wav2spec(target_wav)
            raw_mag, raw_phase = self.audio.wav2spec(raw_wav)
            target_mag = torch.from_numpy(target_mag)
            raw_mag = torch.from_numpy(raw_mag)
            return dvec_mel, target_wav, raw_wav, target_mag, raw_mag, raw_phase
