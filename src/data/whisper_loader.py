#!/usr/bin/env python3

# 2023-2024 Ruchao Fan SPAPL

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset, Audio, concatenate_datasets

class WhisperDataset():
    def __init__(self, data_paths, data_args=None):
        self.data_paths = data_paths
        
        datasets = []
        for split in data_paths.keys():
            if data_args and data_args.use_speed_perturb:
                for sp_rate in np.arange(data_args.sp_low, data_args.sp_high+0.1, 0.1):
                    if sp_rate == 1.0:
                        continue
                    datasets.append(self.create_dataset(data_paths[split], sp_rate=sp_rate))
            
            if data_args and data_args.use_vtlp:
                for vtlp_rate in np.arange(data_args.vtlp_low, data_args.vtlp_high+0.1, 0.1):
                    if vtlp_rate == 1.0:
                        continue
                    datasets.append(self.create_dataset(data_paths[split], vtlp_rate=vtlp_rate))
            
            if data_args and data_args.use_pitch_perturb:
                for n in range(2):
                    datasets.append(self.create_dataset(data_paths[split], pitch_level=data_args.pitch_level))
            
            datasets.append(self.create_dataset(data_paths[split]))
            
        self.data = concatenate_datasets(datasets)
    
    def create_dataset(self, data_path, sp_rate=1.0, vtlp_rate=1.0, pitch_level=0.0):

        audio_dict = self._load_audio_path(data_path['scp_path'])
        label_dict = self._load_label(data_path['text_label'])
        assert len(audio_dict) == len(label_dict), "label and sample size mismatch"

        paired_dict = {"utt_id": [], "audio": [], "sentence": [], "sp_rate": [], "vtlp_rate": [], "pitch_level": []}

        # prepare dictionary for huggingface dataset
        for i in range(len(audio_dict)):
            utt, audio_path = audio_dict[i]
            label = label_dict[utt]
            paired_dict["utt_id"].append(utt)
            paired_dict["audio"].append(audio_path)
            paired_dict["sentence"].append(label)

            paired_dict["sp_rate"].append(sp_rate)
            paired_dict["vtlp_rate"].append(vtlp_rate)

            if pitch_level != 0.0:
                used_pitch_level = np.random.choice(np.arange(-pitch_level, pitch_level+1.0, 1.0))
            else:
                used_pitch_level = pitch_level
                
            paired_dict["pitch_level"].append(used_pitch_level)
        
        dataset = Dataset.from_dict(paired_dict).cast_column("audio", Audio())
        
        return dataset
    
    def _load_audio_path(self, wav_path):
        
        audio_dict = []

        with open(wav_path, 'r') as fin:
            line = fin.readline()
            while line:
                line = line.strip().split(' ')
                utt = line[0]
                i = 1
                
                try:
                    while not os.path.exists(line[i]):
                        i += 1
                except:
                    print("SCP FILE Not valid!")
                    break
                
                audio_dict.append((utt, line[i]))
                line = fin.readline()

        print("Reading {} lines from {}".format(len(audio_dict), wav_path))

        return audio_dict
    
    def _load_label(self, lab_path):
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()
            
            while line:
                line= line.strip().split(' ')
                label_dict[line[0]] = ' '.join(line[1:])    
                line = fin.readline()
        
        print("Reading {} lines from {}".format(len(label_dict), lab_path))
        return label_dict

