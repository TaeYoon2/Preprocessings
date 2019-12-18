# -*- coding:utf-8 -*-
## import #############################################################################################################
import os
import sys

import json
import librosa
import librosa.filters
import numpy as np
import pathlib

import scipy

# pooling & time
# from multiprocessing import Pool
# import time



## Data ###############################################################################################################
class Wav_preprocessing:
    """
    Note:
    
    attributes:
        process_like_copy
        _process_and_save
        _extract_wav_type_features
        _load_wav
        _trim_silences
        _stft
        _mel_filter
        _transpose
        _log_clip
        _to_decibel
        _normalize
        
    """
    ######
    # Init
    ######
    def __init__(self, config):
        super(Wav_preprocessing, self).__init__()

        """
        Note:

        Args:


        Returns:

        """
        self._sample_rate = config['sample_rate']
        self._preemphasis_coef = config['preemphasis_coef']
        self._nfft = config['nfft']
        self._fft_window_size = config['fft_window_size']
        self._fft_hop_size = config['fft_hop_size']
        self._fft_window_fn = config['fft_window_fn']
        self._fft_is_center = config['fft_is_center']
        self._mel_bins = config['mel_bins']
        self._mel_fmin = config['mel_fmin']
        self._mel_fmax = config['mel_fmax']
        self._top_db = config['top_db']
        self._clip_mel = config['clip_mel']
        self._clip_linear = config['clip_linear']
        self._is_legacy = config['is_legacy']
        self._min_db = config['min_db']
        self._max_normalized_db = ['max_normalized_db']
        self._min_normalized_db = ['min_normalized_db']


    def process_like_copy(self, source, target, src_type='.wav', num_pool=4):
        # 디렉토리 구조를 그대로 복사하면서 파일 처리
        # with Pool(num_pool) as pool:
        for (path, _dir, files) in os.walk(source):
            for filename in files:
                base, ext = os.path.splitext(filename)
                if ext == src_type:
                    rel_path = os.path.relpath(path, source)
                    source_file = os.path.join(source, rel_path, filename)
                    target_dir = os.path.join(target, rel_path)
                    target_file = os.path.join(target, rel_path, base+".npy")
                    # 타겟 경로 없을 시 생성
                    if not os.path.exists(target_dir):
                        print("makedirs")
                        os.makedirs(target_dir, exist_ok=True)
                    # 파일 아직 처리 안했으면
                    if not os.path.exists(target_file):
                        self._process_and_save(source_file, target_file)


    def _process_and_save(self, src, target):
        print("_process_save")
        mel = np.empty(shape=[])
        mel , _ = self._extract_wav_type_features(src)
        np.save(target, mel)

    #########################################################
    # Local funcs: components of the process units (wav-type)
    #########################################################

    def _extract_wav_type_features(self, sig):
        """
        Note:
            extract log mel feature [, linear spectrogram feature] from a signal

        Args:
            sig: np array, sound signal

        Returns:
            _mel_spec: np array, computed mel feature output
            _spec: np array, linear (not mel filtered) signal

        """
        _wav = self._trim_silences(sig)
        # self._pre_emphasis()
        _spec = self._stft(_wav)
        _mel_spec = self._mel_filter(_spec)
        _mel_spec, _spec = self._transpose(_mel_spec), self._transpose(_spec)

        if self._is_legacy:
            _mel_spec, _spec = self._normalize(self._to_decibel(_mel_spec)), self._normalize(self._to_decibel(_spec))
        else:
            _mel_spec, _spec = self._log_clip(_mel_spec, self._clip_mel), self._log_clip(_mel_spec, self._clip_linear)

        return _mel_spec, _spec





    ###########################################################################
    # Local funcs: components of the components of the process units (wav-type)
    ###########################################################################

    ### Wav file loader
    def _load_wav(self, filename):
        # Load wav file
        # by librosa
        # self._wav = librosa.core.load(self._WAV_FILE, sr=self._sample_rate)[0]
        # by scipy
        # sr, _wav = scipy.io.wavfile.read(filename)
        # _wav = _wav / 32768.0
        # by numpy
        _wav, sr = np.load(filename)

        del sr

        return _wav

    def _trim_silences(self, wav):
        """
        Note:
            trim leading and trailing silence under self._top_db

        Args:
            wav: np array, audio signal

        Returns:
            _wav: np array, trimmed signal

        """

        _wav, _ = librosa.effects.trim(wav, top_db=self._top_db, frame_length=2048, hop_length=512)
        return _wav

    def _stft(self, wav):
        """
        Note:
            STFT with librosa

        Args:
            wav: np array, sound signal

        Returns:
            _spec: np array, stft result

        """

        # short time fourier transformation
        _spec = librosa.stft(y=wav,
                             n_fft=self._nfft,
                             hop_length=self._fft_hop_size,
                             win_length=self._fft_window_size,
                             window=self._fft_window_fn,
                             center=self._fft_is_center,
                             pad_mode='reflect')
        _spec = np.abs(_spec)**2
        return _spec

    def _mel_filter(self, spec):
        """
        Note:
            apply mel filters

        Args:
            spec: np array, an output of STFT function

        Returns:
            _mel_spec: mel filtered STFT output

        """

        # Pass mel filter banks
        mel_filter_banks = librosa.filters.mel(sr=self._sample_rate, n_fft=self._nfft,
                                               n_mels=self._mel_bins, fmin=self._mel_fmin, fmax=self._mel_fmax)
        _mel_spec = np.dot(mel_filter_banks, spec)
        return _mel_spec

    def _transpose(self, spec):
        """
        Note:
            transpose feature

        Args:
            spec: np array, feature array

        Returns:
            _spec: transposed feature

        """

        _spec = np.transpose(spec, (1, 0))

        return _spec

    def _log_clip(self, spec, clip):
        """
        Note:
            clip by self._clip_mel and transform mel into log mel

        Args:
            spec: np array, computed mel filtered feature

        Returns:
            _log_clipped_spec: np array, log clipped feature

        """

        _clipped_spec = np.clip(spec, clip, None)
        _log_clipped_spec = np.log(_clipped_spec)

        return _log_clipped_spec

    def _to_decibel(self, spec):
        """
        Note:
            convert mel feature to decibel scale

        Args:
            spec: np array, spectrogram(output of stft~)

        Returns:
            _db_spec: np array, spectrogram in decibel

        """

        min_level = np.exp(-100/20*np.log(10))
        _db_spec = 20*np.log10(np.maximum(min_level, spec))-20

        return _db_spec

    def _to_amplitude(self, decibel):
        """
        Note:
            convert decibel scale to origin

        Args:
            decibel: np array, decibel

        Returns:
            _db_mel_spec: np array, mel spectrogram in decibel

        """
        return np.power(10, decibel * (1/20))

    def _normalize(self, db_spec):
        """
        Note:
            normalize spectrogram feature

        Args:
            db_spec: np array, spectrogram in decibel

        Returns:
            _normalized_db_spec: np array, normalized spectrogram in decibel

        """

        _normalized_db_spec = np.clip(2*self._max_normalized_db*((db_spec-self._min_db)/-self._min_db)-self._max_normalized_db, self._min_normalized_db, self._max_normalized_db)

        return _normalized_db_spec


db_root = '/mnt/data1/youngsunhere/data/namz_eng_spkrid/libri/train-clean-460'
save_root = '/mnt/nas_crawled/taeyoon_chestnut/npy_mel/libri/train-clean-460'

if __name__ == "__main__":
    with open("config.json", "rb") as fid:
        config = json.load(fid)
    processor = Wav_preprocessing(config)
    processor.process_like_copy(db_root, save_root, src_type=".npy")
