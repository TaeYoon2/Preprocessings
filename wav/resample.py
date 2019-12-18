import os
import librosa
# import numpy as np
import soundfile as sf
# import matplotlib.pyplot as plt
import multiprocessing

def down_sample(input_wav):
    origin_sr=16000
    resample_sr=8000
    original_data_root = '/mnt/nas/01_ASR/01_korean/01_speech_text/KOR-CLN-V1'
    savedir='/mnt/data1/sunkist/data/KOR-CLN-V1-8k'
    rel_path = os.path.relpath(input_wav, original_data_root)
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, sr, resample_sr)
    print("original wav sr: {}, original wav shape: {}, resample wav sr: {}, resmaple shape: {}".format(origin_sr, y.shape, resample_sr, resample.shape))

    file_path = os.path.join(savedir, rel_path)
    print(file_path)
    sf.write(file_path, resample, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')

def sweap_dir(root, savedir):
    to_be_processed = []
    for items in os.walk(root):
        pardir, _, filenames = items
        for filename in filenames:
            if os.path.splitext(filename)[1] == ".wav":
                original_wav_path = os.path.join(pardir,filename)
                relpath = os.path.relpath(pardir, root)
                each_dir = os.path.join(savedir, relpath)
                if not os.path.exists(each_dir):
                    os.makedirs(each_dir, exist_ok=True)
                to_be_processed.append(original_wav_path)

    pool = multiprocessing.Pool(processes=12)
    pool.map(down_sample, to_be_processed)

original_data_root = '/mnt/nas/01_ASR/01_korean/01_speech_text/KOR-CLN-V1'
savedir = '/mnt/data1/sunkist/data/KOR-CLN-V1-8k'
sweap_dir(original_data_root, savedir)
# down_sample(man_original_data, 16000, 8000, savedir)