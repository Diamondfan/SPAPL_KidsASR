#!/usr/bin/env python3
# 2023-2024 Ruchao Fan  UCLA SPAPL

from transformers import pipeline, WhisperForConditionalGeneration, WhisperTokenizer
from transformers import WhisperFeatureExtractor
import os
from datasets import Dataset, load_dataset, Audio
import soundfile as sf
import json
# from whisper_normalizer.english import EnglishTextNormalizer
from collections import defaultdict
from evaluate import load
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from Rishabh_norm import RishabhTextNormalizer

def main():
    parser = argparse.ArgumentParser(description="transcribe the input wav")

    parser.add_argument("--wav_path")
    parser.add_argument("--lm_config")
    parser.add_argument("--data_path")
    parser.add_argument("--local_rank", default=0, type=int, help="model parallel decoding with deepspeed")
    parser.add_argument("--text_label", default="", type=str, help='text label')
    parser.add_argument("--utt2num_frames", default="", type=str, help='utt2frames to filter too long utterances')
    parser.add_argument("--task", default="art", type=str, help="the task for training, art, cassnat, ctc")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--load_data_workers", default=1, type=int, help="Number of parallel data loaders")
    parser.add_argument("--resume_model", default='', type=str, help="Model to do evaluation")
    parser.add_argument("--result_file", default='', type=str, help="File to save the results")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--rnnlm", type=str, default=None, help="RNNLM model file to read")
    parser.add_argument("--rank_model", type=str, default='lm', help="model type to Rank ESA")
    parser.add_argument("--lm_weight", type=float, default=0.1, help="RNNLM weight")
    parser.add_argument("--seed", default=1, type=int, help="random number seed")


def load_data(dataset):
    datasets = []
    for ds in os.listdir("./data/test"):
        if dataset != "all":
            if ds != dataset:
                continue
        dataset = Dataset.load_from_disk("./data/test/" + ds)
        dataset.name = ds

        dataset = dataset.cast_column("audio_path", Audio())
        dataset = dataset.rename_column("audio_path", "audio")
        datasets.append(dataset)
    print("Dataset prepared")
    print(f"Found {len(datasets)} datasets")
    for dataset in datasets:
        assert "input_features" in dataset.column_names, "input_features not in dataset"
        assert "labels" in dataset.column_names, "labels not in dataset"
        print(f"Found {len(dataset)} test examples in {dataset.name}")    
    return datasets



def transcribe(model_name, dataset="all"):
    if "small" in model_name:
        base_model = "openai/whisper-small"
    elif "medium" in model_name:
        base_model = "openai/whisper-medium"
    elif "large-v2" in model_name:
        base_model = "openai/whisper-large-v2"
        
    else:
        base_model = model_name
        # raise ValueError("Model name must contain either small.en or medium.en")
    if ".en" in model_name and ".en" not in base_model:
        base_model += ".en"
    print(f"Base model: {base_model}")
    if "rishabh" in model_name.lower():
        normalizer = RishabhTextNormalizer
        base_model = model_name
        tokenizer = WhisperTokenizer.from_pretrained(base_model, language="english", task="transcribe")
        
    else:
        tokenizer = WhisperTokenizer.from_pretrained(base_model, language="english", task="transcribe")
        normalizer = tokenizer._normalize
        
    metric = load("wer")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    pipe = pipeline(task = "automatic-speech-recognition", model=model, tokenizer=base_model, feature_extractor=WhisperFeatureExtractor.from_pretrained(base_model), device="cuda", chunk_length_s=30,)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Base model: {base_model}")
        
    if dataset == "librispeech":
        testsets = [load_dataset("librispeech_asr", "clean", split="test")]
        testsets[0].name = "librispeech"
        print(testsets[0])
    else:
        testsets = load_data(dataset)
    def transcribe_file(audio):
        text = pipe(audio)["text"]
        return text
    transcription_dir = model_name + "/transcriptions"
    if "fine-tuned-whisper" not in transcription_dir:
        transcription_dir = "huggingface_models_transcription/" + transcription_dir
    os.makedirs(transcription_dir, exist_ok=True)
    for testset in testsets:
        datasets = {"ground_truths": [], "hypotheses": []}
        print(f"Transcribing {testset.name}")
        transcription_file = transcription_dir + "/" + testset.name + ".txt"
        transcription_file = open(transcription_file, "w")
        for out, line in tqdm(zip(pipe(KeyDataset(testset, "audio")), testset), desc=f"Transcribing {testset.name}", total=len(testset)):

            transcription = out["text"]
            if testset.name == "librispeech":
                ground_truth = line["text"]
            else:
                ground_truth = line["sentence"]
            path = line["audio"]["path"]
            datasets["ground_truths"].append(normalizer(ground_truth))
            datasets["hypotheses"].append(normalizer(transcription))
            transcription_file.write(path + "\t" + transcription + "\n")
        wer = metric.compute(predictions=datasets["hypotheses"], references=datasets["ground_truths"]) * 100
        print("Dataset: {} WER: {}".format(testset.name, wer))
    
if __name__ == "__main__":
    main()
