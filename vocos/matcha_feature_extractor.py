import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

from vocos.feature_extractors import FeatureExtractor


class MatchaMelSpectrogramFeatures(FeatureExtractor):
    """
    Generate MelSpectrogram from audio using same params
    as Matcha TTS (https://github.com/shivammehta25/Matcha-TTS)
    This is also useful with tacatron, waveglow..etc.
    """

    def __init__(
        self,
        *,
        mel_mean,
        mel_std,
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        n_mels=80,
        hop_length=256,
        center=False,
        f_min=0,
        f_max=8000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.f_min = f_min
        self.f_max = f_max
        # Data-dependent
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        # Cache
        self._mel_basis = {}
        self._hann_window = {}

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        mel = self.mel_spectrogram(audio).squeeze()
        mel = normalize(mel, self.mel_mean, self.mel_std)
        return mel.unsqueeze(0)

    def mel_spectrogram(self, y):
        mel_basis_key = str(self.f_max) + "_" + str(y.device)
        han_window_key = str(y.device)
        if mel_basis_key not in self._mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max
            )
            self._mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)
            self._hann_window[han_window_key] = torch.hann_window(self.win_length).to(y.device)
        pad_vals = (
            (self.n_fft - self.hop_length) // 2,
            (self.n_fft - self.hop_length) // 2,
        )
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            pad_vals,
            mode="reflect"
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann_window[han_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self._mel_basis[mel_basis_key], spec)
        spec = spectral_normalize_torch(spec)
        return spec


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def normalize(data, mu, std):
    if not isinstance(mu, (float, int)):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, (float, int)):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return (data - mu) / std
