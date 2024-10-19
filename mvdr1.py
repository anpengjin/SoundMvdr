#encoding:utf-8
import os

import torch
import torchaudio
import torch.nn as nn
import torchaudio.functional as F
import soundfile as sf

from utils import audio_read, TorchSTFT

REFERENCE_CHANNEL = 0
eps = 1e-16

def get_irms(stft_clean, stft_noise):
    mag_clean = stft_clean.abs() ** 2
    mag_noise = stft_noise.abs() ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise)
    # irm_noise = mag_noise / (mag_clean + mag_noise)
    return irm_speech[REFERENCE_CHANNEL]

class SoundMvdr(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, stft_mix, stft_mask):
        '''
        stft_mix(complex): [B, C, F, T]
        stft_mask(float): [B, 1, F, T]
        '''
        print(stft_mix.shape, stft_mask.shape)

        B, C, F, T = stft_mix.shape

        speech_psd = torch.zeros((B, F, C, C))
        noise_psd = torch.zeros((B, F, C, C))

        mvdr_out = []

        for i in range(T):
            spec = stft_mix[..., i].permute(0, 2, 1).unsqueeze(-1)  # [B, F, C, 1]
            spec_psd = torch.matmul(spec, spec.permute(0, 1, 3, 2).conj())
            print('spec_psd', spec_psd.shape)  # [B, F, C, C]

            smooth_factor = 0.9 if i < 100 else 0.98
            spec_mask = stft_mask[:, 0, :, i].unsqueeze(-1).unsqueeze(-1)
            
            # speech_psd = smooth_factor * speech_psd + (1 - smooth_factor) * spec_psd * spec_mask
            # noise_psd = smooth_factor * noise_psd + (1 - smooth_factor) * spec_psd * (1 - spec_mask)
            ax = smooth_factor + (1 - spec_mask) * (1 - smooth_factor)
            an = smooth_factor + spec_mask * (1 - smooth_factor)
            speech_psd = ax * speech_psd + (1 - ax) * spec_psd
            noise_psd  = an * noise_psd + (1 - an) * spec_psd
            
            print('speech_psd', speech_psd.shape)  # [B, F, C, C]
            print('noise_psd', noise_psd.shape)  # [B, F, C, C]

            noise_psd_inv = noise_psd.clone()
            noise_psd_inv[:, :, 0, 0] = noise_psd[:, :, -1, -1]
            noise_psd_inv[:, :, -1, -1] = noise_psd[:, :, 0, 0]
            noise_psd_inv[:, :, 1, 0] = -noise_psd[:, :, 1, 0]
            noise_psd_inv[:, :, 0, 1] = -noise_psd[:, :, 0, 1]  # [B, F, C, C]

            numerator = torch.matmul(speech_psd, noise_psd_inv)
            trace = torch.diagonal(numerator, 0, dim1=-1, dim2=-2).sum(dim=-1)
            ws = numerator / (trace[..., None, None] + eps)
            beamform_weights = ws[..., :, REFERENCE_CHANNEL]

            specgram_enhanced = torch.einsum("...fc,...fct->...ft", [beamform_weights.conj(), spec])
            mvdr_out.append(specgram_enhanced)

        mvdr_out = torch.cat(mvdr_out, dim=-1)

        return mvdr_out

        pass


if __name__ == "__main__":
    # 计算mask
    wav_path = f'.\data\waveform_clean.wav'
    signal = audio_read(wav_path)
    print('signal', signal.shape)

    stft_fun = TorchSTFT()
    signal_clean_stft = stft_fun.fft(signal)
    print(signal_clean_stft.shape)  # torch.Size([8, 257, 401]) torch.complex

    wav_path = f'.\data\\noise.wav'
    signal = audio_read(wav_path)
    print('signal', signal.shape)

    stft_fun = TorchSTFT()
    signal_noise_stft = stft_fun.fft(signal)
    print(signal_noise_stft.shape)  # torch.Size([8, 257, 401]) torch.complex

    stft_mask = get_irms(signal_clean_stft, signal_noise_stft)
    print('stft_mask', stft_mask.shape)  # torch.Size([257, 401])

    # 得到多麦信号
    wav_path = f'.\data\waveform_mix.wav'
    signal = audio_read(wav_path)
    print('signal', signal.shape)

    stft_fun = TorchSTFT()
    signal_mix_stft = stft_fun.fft(signal)
    print(signal_mix_stft.shape)  # torch.Size([8, 257, 401]) torch.complex

    # MVDR 前向步骤
    mvdr_fun = SoundMvdr()
    mvdr_out = mvdr_fun(signal_mix_stft.unsqueeze(0), stft_mask.unsqueeze(0).unsqueeze(0))
    print('mvdr_out', mvdr_out.shape)  # torch.Size([1, 257, 401])

    signal = stft_fun.ifft(mvdr_out)
    print('mvdr_out_time', signal.shape)  # torch.Size([1, 64000])

    out_path = f'.\data\sound_mvdr2.wav'
    sf.write(out_path, signal.T, 16000)
