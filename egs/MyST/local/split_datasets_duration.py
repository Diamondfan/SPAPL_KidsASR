#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Split the datasets into a short and long one according to a duration threshold
# return uttlist

import os
import sys
import argparse

sys.path.append(os.environ['rootdir']+'/src')
from data.whisper_loader import WhisperDataset 


def main():
    parser = argparse.ArgumentParser(description="split_datasets based on duration")
    parser.add_argument("--raw_dataset", required=True, type=str)
    parser.add_argument("--dur_threshold", required=True, type=str)
    parser.add_argument("--short_list", required=True, type=str)
    parser.add_argument("--long_list", required=True, type=str)

    args = parser.parse_args()

    wav_scp = os.path.join(args.raw_dataset, "wav.scp")
    trn_scp = os.path.join(args.raw_dataset, "text")
    data_path = {"data": {"scp_path": wav_scp, "text_label": trn_scp}}
    dataset = WhisperDataset(data_path).data

    num_utt = 0
    with open(args.short_list, 'w') as short_wf, open(args.long_list, 'w') as long_wf:
        for testdata in dataset:
            num_utt += 1

            audio = testdata["audio"]
        
            audio_duration = len(audio["array"]) / audio["sampling_rate"]
            if audio_duration < float(args.dur_threshold):
                short_wf.write(testdata["utt_id"] + "\n")
            else:
                long_wf.write(testdata["utt_id"] + "\n")

            if num_utt % 1000 == 0:
                print("Processed {} utterances out of {}".format(num_utt, len(dataset)), flush=True)

    print("Done, the uttlist are saved in {} and {}".format(args.short_list, args.long_list))

if __name__ == "__main__":
    main()

