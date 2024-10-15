from typing import List, Optional

import torch
import torchaudio
from torch import nn
import math
from decoder.modules import safe_log
from encoder.modules import SEANetEncoder
from encoder.quantization import MScaleRVQ


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class MultiScaleFeatures(FeatureExtractor):
    def __init__(
        self,
        dimention: int = 512,
        ratios: List[int] = [8, 5, 4, 2],
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        kmeans_init: bool = True,
        quantize_dropout: bool = False,
        rand_num_quant: Optional[List] = None,
        vq_scales: Optional[List[int]] = [8, 4, 2, 1],
        ema_decay: float = 0.95,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.rand_num_quant = rand_num_quant
        self.quantize_dropout = quantize_dropout

        self.encoder = SEANetEncoder(
            causal=False,
            n_residual_layers=1,
            norm="weight_norm",
            pad_mode="reflect",
            lstm=2,
            dimension=dimention,
            channels=1,
            n_filters=32,
            ratios=ratios,
            activation="ELU",
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            true_skip=False,
            compress=2,
        )

        self.quantizer = MScaleRVQ(
            dimension=dimention,
            n_q=num_quantizers,
            bins=codebook_size,
            kmeans_init=kmeans_init,
            vq_scales=vq_scales,
            quantize_dropout=quantize_dropout,
            rand_num_quant=rand_num_quant,
            decay=ema_decay,
        )

    def forward(self, audio: torch.Tensor, bandwidth_id: torch.Tensor):
        audio = audio.unsqueeze(1)
        emb = self.encoder(audio)
        rand_quantize_dropout_index = self.rand_num_quant[bandwidth_id] if bandwidth_id is not None else None
        q_res = self.quantizer(emb, self.sample_rate, rand_quantize_dropout_index=rand_quantize_dropout_index)
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty

        return quantized, codes, commit_loss

    def infer(self, audio: torch.Tensor, bandwidth_id: torch.Tensor):
        audio = audio.unsqueeze(1)
        emb = self.encoder(audio)
        rand_quantize_dropout_index = self.rand_num_quant[bandwidth_id] if bandwidth_id is not None else None
        q_res = self.quantizer(emb, self.sample_rate, rand_quantize_dropout_index=rand_quantize_dropout_index)
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty  # codes(8,16,75),features(16,128,75)

        return quantized, codes, commit_loss


if __name__ == "__main__":
    import yaml
    device = "cuda"
    audio = torch.randn(2, 25600).to(device)

    config = yaml.load(open("config/snac_vocos_nq4_scale8421_16khz.yaml", "r"), yaml.Loader)

    print(config["model"]["init_args"]["feature_extractor"]["init_args"])
    mscodec = MultiScaleFeatures(**config["model"]["init_args"]["feature_extractor"]["init_args"]).to(device)
    quantized, codes, commit_loss = mscodec(audio, torch.tensor(0).to(device))

    print(quantized.shape)
    print(codes)
    print(commit_loss)

    for code in codes:
        print(code.shape)
