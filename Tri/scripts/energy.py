
import numpy as np
import pyworld as pw
import soundfile as sf
import librosa
import python_speech_features as psf
import math
import wave

class AudioProcesser:
    def __init__(self, wav_path, hop_size):
        self.hop_size = hop_size
        # self.wav_data, self.sr = librosa.load(wav_path, 16000)
        self.wav_data, self.sr = sf.read(wav_path)
        # make sure input 16kHz audio
        assert self.sr == 16000
        fw = wave.open(wav_path, 'r')
        params = fw.getparams()
        nchannels, sampwidth, self.framerate, self.nframes = params[:4]
        strData = fw.readframes(self.nframes)
        self.waveData = np.fromstring(strData, dtype=np.int16)

    
    def get_pitch(self, eps=1e-5, log=True, norm=True):
        _f0, t = pw.dio(self.wav_data, self.sr, frame_period=self.hop_size / self.sr * 1000)  # raw pitch extractor
        f0 = pw.stonemask(self.wav_data, _f0, t, self.sr)  # pitch refinement
        if log:
            f0 = np.log(np.maximum(eps, f0))
        if norm:
            f0 = (f0 - f0.mean()) / f0.std()
        
        return f0

    def wav2mel(self, fft_size=1024, hop_size=256, 
                win_length=1024, win_mode="hann",
                num_mels=80, fmin=80,
                fmax=7600, eps=1e-10):
        # get amplitude spectrogram
        x_stft = librosa.stft(self.wav_data, n_fft=fft_size, hop_length=hop_size,
            win_length=win_length, window=win_mode, pad_mode="constant")
        spc = np.abs(x_stft)  # (n_bins, T)

        # get mel basis
        fmin = 0 if fmin == -1 else fmin
        fmax = self.sr / 2 if fmax == -1 else fmax
        mel_basis = librosa.filters.mel(self.sr, fft_size, num_mels, fmin, fmax)
        mel = mel_basis @ spc
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)

        return mel.T
    
    def get_energy(self):
        """Extract energy same with FastSpeech2
        """
        mel = self.wav2mel(hop_size=self.hop_size)
        energy = np.sqrt((np.exp(mel) ** 2).sum(-1))

        return energy
    
    def get_energy_psf(self):
        """Directly use python package python_speech_features to extract pitch.
        Can be compared with self.get_energy(), which one is better.
        """
        # python_speech_features use scipy.io.wavfile to read audio
        # so the returned audio data ranges between (-2^15, 2^15) 
        wav_data = self.wav_data * (2**15)
        fbank, energy = psf.fbank(wav_data, samplerate=self.sr, winstep=self.hop_size / self.sr)
        energy = np.log(energy)

        return fbank, energy
    
    # def get_ppg(self):
    #     return self.ppg_extractor.extract_ppg_from_sentence(self.wav_data, self.sr)

    # method 1: absSum
    def calVolume(self, frameSize=256, overLap=128):
        waveData = self.waveData * 1.0 / max(abs(self.waveData))
        wlen = len(waveData)
        step = frameSize - overLap
        frameNum = int(math.ceil(wlen * 1.0 / step))
        volume = np.zeros((frameNum, 1))
        for i in range(frameNum):
            curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
            curFrame = curFrame - np.median(curFrame)  # zero-justified
            volume[i] = np.sum(np.abs(curFrame))
        return volume


    # method 2: 10 times log10 of square sum
    def calVolumeDB(self, frameSize=256, overLap=128):
        # waveData = librosa.util.normalize(self.waveData)
        waveData = self.waveData * 1.0 / max(abs(self.waveData))  # normalization
        wlen = len(waveData)
        step = frameSize - overLap
        frameNum = int(math.ceil(wlen * 1.0 / step))
        volume = np.zeros((frameNum, 1))
        for i in range(frameNum):
            curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
            curFrame = curFrame - np.mean(curFrame)  # zero-justified
            volume[i] = 10 * np.log10(np.sum(curFrame * curFrame))
        return volume

