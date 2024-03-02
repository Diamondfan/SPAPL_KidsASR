#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Compute the duration in hours of a given dataset

import os
import sys
import argparse

sys.path.append(os.environ['rootdir']+'/src')
from data.whisper_loader import WhisperDataset 


def main():
    parser = argparse.ArgumentParser(description="compute duration of a given dataset")
    parser.add_argument("--wav_scp", required=True, type=str)
    parser.add_argument("--trn_scp", required=True, type=str)
    parser.add_argument("--write_utt2dur", required=True, type=str)

    args = parser.parse_args()

    data_path = {"data": {"scp_path": args.wav_scp, "text_label": args.trn_scp}}
    dataset = WhisperDataset(data_path).data

    num_utt = 0
    total_duration = 0

    utt2dur_file = open(args.write_utt2dur, "w")

    for testdata in dataset:
        num_utt += 1
        audio = testdata["audio"]
        uttid = testdata["utt_id"]
        
        audio_duration = len(audio["array"]) / audio["sampling_rate"]
        utt2dur_file.write(uttid + " " + str(audio_duration) + "\n")
        total_duration += audio_duration

        if num_utt % 1000 == 0:
            print("Processed {} utterances out of {}".format(num_utt, len(dataset)), flush=True)
    
    utt2dur_file.close()
    total_duration = total_duration / 3600
    print("Total duration of dataset {} is {} hours".format(args.wav_scp, total_duration))

if __name__ == "__main__":
    main()

