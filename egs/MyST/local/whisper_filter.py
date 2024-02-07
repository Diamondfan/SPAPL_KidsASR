#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The filtering rule is based on paper https://arxiv.org/pdf/2309.07927.pdf
# 1. Filter out the utterances with WER < threshold (50%)
# 2. Filter out the utterances with less than n (3) words  
# 3. Remove utterances longer than 30 seconds in training and development
# 4. Normalize text {<no_signal> digits into to the same format}

import os
import sys
import argparse
import torch
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.append(os.environ['rootdir']+'/src')
from data.whisper_loader import WhisperDataset 


def main():
    parser = argparse.ArgumentParser(description="loading whisper model to filter the utterance with low quality transcription")
    parser.add_argument("--wav_scp", required=True, type=str)
    parser.add_argument("--trn_scp", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--wer_threshold", default=0.5, type=float)
    parser.add_argument("--remove_n_words", required=True, default=3, type=int)
    parser.add_argument("--remove_long_dur", required=True, default=30, type=int)
    parser.add_argument("--saved_utt_list", required=True, type=str)
        
    args = parser.parse_args()

    data_path = {"data": {"scp_path": args.wav_scp, "text_label": args.trn_scp}}
    dataset = WhisperDataset(data_path).data

    processor = WhisperProcessor.from_pretrained(args.model, cache_dir="cached_whisper_models/")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, cache_dir="cached_whisper_models/").to("cuda")
    
    print("Model Loaded, Start Decoding and do filtering...")
    metric_wer = evaluate.load("wer")
    uttlist_writer = open(args.saved_utt_list, "w")
    uttlist_writer.flush()

    num_utt = 0
    for testdata in dataset:
        num_utt += 1
        audio = testdata["audio"]

        if args.remove_long_dur > 0:
            audio_duration = len(audio["array"]) / audio["sampling_rate"]
            if audio_duration > args.remove_long_dur:
                continue 
        
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        reference = processor.tokenizer._normalize(testdata['sentence'])
 
        if len(reference.split(" ")) < args.remove_n_words:
            continue

        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        wer = metric_wer.compute(references=[reference], predictions=[prediction])

        if wer > args.wer_threshold:
            continue

        if num_utt % 1000 == 0:
            print("Processed {} utterances out of {}".format(num_utt, len(dataset)), flush=True)

        uttlist_writer.write(testdata["utt_id"] + "\n")
    
    uttlist_writer.close()


if __name__ == "__main__":
    main()

