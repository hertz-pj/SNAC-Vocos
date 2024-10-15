from dataclasses import dataclass

import numpy as np
import torch
import torch.utils
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import Dataset, Audio

import soundfile
# import librosa

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class HFDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = HFDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path, encoding="utf-8") as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.hf_dataset = Dataset.from_dict(
            {"audio": self.filelist, "text": self.filelist}
        ).cast_column("audio", Audio())

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx):
        audio = self.hf_dataset[idx]["audio"]
        y1 = torch.tensor(audio["array"]).float().unsqueeze(0)
        sr = audio["sampling_rate"]

        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y1, sr, [["norm", f"{gain:.2f}"]])

        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]


