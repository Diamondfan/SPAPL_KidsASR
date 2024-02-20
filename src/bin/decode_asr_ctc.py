#!/usr/bin/env python3
# 2023-2024 Ruchao Fan  UCLA SPAPL

# decode the entire test sets with the huggingface model
import os
import sys
import argparse
import torch
import evaluate
from transformers import AutoProcessor, AutoConfig, AutoModelForCTC

sys.path.append(os.environ['rootdir']+'/src')
from data.whisper_loader import WhisperDataset 
from arguments import PEFTArguments
from data.english_normalizer import EnglishTextNormalizer

def main():
    parser = argparse.ArgumentParser(description="Decoding the evaluation data in the wav_scp file")
    parser.add_argument("--wav_scp", required=True, type=str)
    parser.add_argument("--trn_scp", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--processor", required=True, type=str)
    parser.add_argument("--result_ref_file", required=True, type=str)
    parser.add_argument("--result_hyp_file", required=True, type=str)
    parser.add_argument("--compute_wer", required=True, default=True, type=bool)
    parser.add_argument("--chunk_length", required=True, default=30, type=int)
        
    args = parser.parse_args()

    data_path = {"data": {"scp_path": args.wav_scp, "text_label": args.trn_scp}}
    dataset = WhisperDataset(data_path).data

    print("Loading Model....")
    cache_dir_processor = "cached_whisper_models/" if not os.path.exists(args.processor) else None
    cache_dir_model = "cached_whisper_models/" if not os.path.exists(args.model) else None 
    
    config = AutoConfig.from_pretrained(args.model, cache_dir=cache_dir_model)
    
    processor = AutoProcessor.from_pretrained(args.processor, cache_dir=cache_dir_processor)
    
    if hasattr(config, "peft_config"): 
        config.peft_config = PEFTArguments(**config.peft_config)
    
    model = AutoModelForCTC.from_pretrained(args.model, config=config, cache_dir=cache_dir_model).to("cuda")

    if hasattr(config, "peft_config"):
        if config.peft_config.peft_type == "prefix_tuning":
            model.generation_config.max_length -= config.peft_config.prefix_seq_len[1]
    
    num_utt = 0
    if args.compute_wer:
        metric_wer = evaluate.load("wer")
        references = []
        transcriptions = []

    ref_writer = open(args.result_ref_file, 'w')
    hyp_writer = open(args.result_hyp_file, 'w')

    normalize_file = os.path.join(args.processor, "normalizer.json")
    with open(normalize_file, encoding="utf-8") as vocab_handle:
        import json
        english_spelling_normalizer = json.load(vocab_handle)
    normalizer = EnglishTextNormalizer(english_spelling_normalizer)

    for testdata in dataset:
        num_utt += 1

        audio = testdata["audio"]
        
        inputs = processor(audio["array"], return_tensors="pt", sampling_rate=audio["sampling_rate"],) #use_vtlp=True,)
        input_features = inputs.input_values.to("cuda")
        with torch.no_grad():
            logits = model(input_features).logits
        
        pred_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(pred_ids)[0]
        
        # todo: beam search decoing with external language model

        reference = normalizer(testdata['sentence'])
        prediction = normalizer(prediction)

        if args.compute_wer and len(reference) > 0:
            references.append(reference)
            transcriptions.append(prediction)

        utt_id = testdata["utt_id"].replace('-', '_')
        ref_writer.write(reference.upper() + ' (' + utt_id + ')\n')
        hyp_writer.write(prediction.upper() + ' (' + utt_id + ')\n')
        
        if num_utt % 200 == 0:
            print("Processed {} utterances out of {}".format(num_utt, len(dataset)), flush=True)
    
    if args.compute_wer:
        wer = metric_wer.compute(references=references, predictions=transcriptions)
        print("Word Error Rate: {}".format(wer), flush=True)

    ref_writer.close()
    hyp_writer.close()
        
if __name__ == "__main__":
    main()
