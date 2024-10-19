#encoding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()

def evaluate(estimate, reference):
    si_snr_score = si_snr(estimate, reference)
    (
        sdr,
        _,
        _,
        _,
    ) = mir_eval.separation.bss_eval_sources(reference.numpy(), estimate.numpy(), False)
    pesq_mix = pesq(SAMPLE_RATE, estimate[0].numpy(), reference[0].numpy(), "wb")
    stoi_mix = stoi(reference[0].numpy(), estimate[0].numpy(), SAMPLE_RATE, extended=False)
    print(f"SDR score: {sdr[0]}")
    print(f"Si-SNR score: {si_snr_score}")
    print(f"PESQ score: {pesq_mix}")
    print(f"STOI score: {stoi_mix}")

class STFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, win_length=480):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.add_window = np.hanning(win_length)
        self.del_window = np.hanning(win_length)

        self.device = 'cpu'

    def enframe(self, signal, frame_len, frame_shift, win):
        """
        calculate the number of frames: 
        frames = (num_samples -frame_len) / frame_shift +1
        """
        num_samples = signal.size
        num_frames = np.floor((num_samples - frame_len) / frame_shift)+1  
        # calculate the numbers of frames
        frames = np.zeros((int(num_frames),frame_len))   # (num_frames,frame_len)
        # Initialize an array for putting the frame signals into it
        for i in range(int(num_frames)):
            frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len]
            frames[i,:] = frames[i,:] * win
        return frames

    def fft(self, signal):
        '''
        signal: [time]
        '''
        begin_zero = np.zeros(self.hop_length * 2)
        signal = np.append(begin_zero, signal)

        frames = self.enframe(signal, self.win_length, self.hop_length, self.add_window)
        # print(frames.shape)

        if self.n_fft > self.win_length:
            fft_zero = np.zeros((frames.shape[0], self.n_fft - self.win_length))
            frames = np.concatenate((frames, fft_zero), axis=1)
            # print(frames.shape)

        spec = np.fft.rfft(frames, self.n_fft)
         
        return torch.view_as_real(torch.tensor(spec)).to(self.device)

    def ifft(self, signal_fft):
        '''待更新'''

class TorchSTFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, win_length=480):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.add_window = torch.hann_window(win_length)
        self.del_window = torch.hann_window(win_length)

        self.device = 'cpu'

    def fft(self, signal):
        '''
        input(torch.float): [B, T]
        output(torch.complex): [B, F, T]
        '''
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal)

        spec = torch.stft(signal, 
                          n_fft=self.n_fft, 
                          hop_length=self.hop_length, 
                          win_length=self.win_length,
                          window=self.add_window,
                          return_complex=True,
                          )
        return spec
 
    def ifft(self, signal_spec):
        '''
        input(torch.complex): [B, F, T]
        output(torch.float): [B, T]
        '''
        signal = torch.istft(signal_spec, 
                             n_fft=self.n_fft, 
                             hop_length=self.hop_length,
                             win_length=self.win_length,
                             window=self.del_window,
                             return_complex=False,
                             )
        return signal

def generate_mixture(waveform_clean, waveform_noise, target_snr):
    power_clean_signal = waveform_clean.pow(2).mean()
    power_noise_signal = waveform_noise.pow(2).mean()
    current_snr = 10 * torch.log10(power_clean_signal / power_noise_signal)
    waveform_noise *= 10 ** (-(target_snr - current_snr) / 20)
    return waveform_clean + waveform_noise

def audio_read(audio_path):
    signal, sr = sf.read(audio_path)
    signal = signal.T
    return signal

if __name__ == "__main__":
    wav_path = f'D:\learning\mvdr\soudnMvdr\data\waveform_clean.wav'
    signal, sr = sf.read(wav_path)
    signal = signal.T
    print(signal.shape, sr)  # (8, 64000) 16000

    stft_fun = TorchSTFT()
    signal_stft = stft_fun.fft(signal)
    print(signal_stft.shape)  # torch.Size([8, 257, 401]) torch.complex

    signal = stft_fun.ifft(signal_stft)
    print(signal.shape)  # torch.Size([8, 64000])

    out_path = f'D:\learning\mvdr\soudnMvdr\data\waveform_clean_fft_ifft.wav'
    sf.write(out_path, signal.T, sr)


