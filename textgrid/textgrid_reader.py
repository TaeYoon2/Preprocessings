# -*- coding:utf-8 -*-
import os
# textgrid tool for praat
import tgt
import librosa
import numpy as np
import glob


def parse_grid(grid_path):
    '''
       note : parse a textgrid and then split the wav matches the textgrid

       arg :
           grid_path : textgrid path to parse
	'''

    dir_path = os.path.dirname(grid_path)
    grid_filename = os.path.basename(grid_path)
    name = os.path.splitext(grid_filename)[0]
    wav_path = os.path.join(dir_path, name+".wav")
    target_tier = "comma"

    tg_obj = tgt.read_textgrid(grid_path)
    # get objects(textgrid-tier & wav) matches the grid path
    tier_obj = tg_obj.get_tier_by_name(target_tier)
    wav_obj, sr = librosa.load(wav_path, sr=None)

    for idx in range(len(tier_obj)):
        part = tier_obj[idx]
        time_s = librosa.time_to_samples(part.start_time, sr)
        time_e = librosa.time_to_samples(part.end_time, sr)
        librosa.output.write_wav('{}_{}.wav'.format(name,idx),
                                 wav_obj[time_s:time_e],
                                 sr)
        with open("{}_{}.txt".format(name,idx), "w") as f:
            f.write(part.text)

if __name__ == "__main__":
    path_type = "/Users/taeyun/Downloads/nk_from_ug/F2/*.TextGrid"
    save_path = "./"
    tg_list = glob.glob(path_type)
    for tg_path in tg_list:
    	parse_grid(tg_path)

