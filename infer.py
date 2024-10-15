import math
import typing as tp
import torchaudio
import torch
import yaml

from encoder.utils import convert_audio
from decoder.pretrained import SnacVocos


class SnacInfer:
    def __init__(self, config_path, model_path, device):
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        self.model = SnacVocos.from_pretrained(config_path, model_path)
        self.model = self.model.to(device)
        self.device = device
        self.hop_length = self.config["model"]["init_args"]["head"]["init_args"]["hop_length"]
        self.vq_scales = self.config["model"]["init_args"]["feature_extractor"]["init_args"]["vq_scales"]

    def preprocess(self, wav):
        length = wav.shape[-1]
        pad_to = self.hop_length * self.vq_scales[0]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        wav = torch.nn.functional.pad(wav, (0, right_pad))
        return wav

    def encode_infer(self, wav, bandwidth_id):
        wav = self.preprocess(wav)
        wav = wav.to(self.device)
        features, discrete_code = self.model.encode_infer(wav, bandwidth_id=bandwidth_id)
        return features, discrete_code

    def codes_to_features(self, codes: tp.List[int]) -> torch.Tensor:
        features = self.model.feature_extractor.quantizer.decode(codes)
        return features

    def decode(self, features, bandwidth_id):
        bandwidth_id.to(self.device)
        audio_out = self.model.decode(features, bandwidth_id=bandwidth_id)
        return audio_out

    def run(self, wav_path, target_sr):
        wav, sr = torchaudio.load(wav_path)
        wav = convert_audio(wav, sr, target_sr, 1)
        wav = self.preprocess(wav)
        bandwidth_id = torch.tensor([3]).to(device)
        wav = wav.to(device)
        features, discrete_code = self.encode_infer(wav, bandwidth_id=bandwidth_id)
        audio_out = self.decode(features, bandwidth_id=bandwidth_id)

        return audio_out


if __name__ == "__main__":
    device = torch.device("cuda")
    config_path = "path/to/config.yaml"
    model_path = "path/to/xxxx.ckpt"

    wav_outpath = "wav_out.wav"
    wav_path = "path/to/xxx.wav"

    wav, sr = torchaudio.load(wav_path)
    wav = convert_audio(wav, sr, 16000, 1)
    bandwidth_id = torch.tensor([3]).to(device)

    snac_infer = SnacInfer(config_path, model_path, device)

    # Reconstruct audio from raw wav
    audio_out = snac_infer.run(wav_path, 16000)
    torchaudio.save(wav_outpath, audio_out.cpu(), 16000, encoding="PCM_S", bits_per_sample=16)

    # Generate discrete codes
    features, codes = snac_infer.encode_infer(wav, bandwidth_id)
    print(features.shape)
    # Note that codes is a list of token sequences of variable lengths,
    # each corresponding to a different temporal resolution.
    for code in codes:
        print(code.shape)

    # Audio reconstruction through codes
    wav_out = snac_infer.decode(features, bandwidth_id)
    print(wav_out.shape)