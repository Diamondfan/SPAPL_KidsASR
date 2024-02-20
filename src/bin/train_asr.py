#!/usr/bin/env python3
# 2023-2024 Ruchao Fan  UCLA SPAPL

# Finetuning loaded huggingface models including Whisper and WavLM
# Reference code: 
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py


import logging
import os
import sys

import datasets
import evaluate
import torch
from datasets import DatasetDict, IterableDatasetDict
from datasets.distributed import split_dataset_by_node

import transformers
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint, is_main_process

sys.path.append(os.environ['rootdir']+'/src') 
from arguments import WhisperModelArguments, DataTrainingArguments, PEFTArguments
from data.whisper_loader import WhisperDataset 
from data.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from models.modeling_whisper import WhisperForConditionalGeneration
from models.feature_extractor import WhisperFeatureExtractor

logger = logging.getLogger(__name__)

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((WhisperModelArguments, DataTrainingArguments, PEFTArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2:
        if sys.argv[1].endswith(".json"):
            model_args, data_args, peft_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        elif sys.argv[1].endswith(".yaml"):
            model_args, data_args, peft_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.warning(f"checkpoint find, will load training from the checkpoint")
        
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    raw_datasets["train"] = WhisperDataset(data_args.train_data_path, data_args).data
    raw_datasets["valid"] = WhisperDataset(data_args.dev_data_path).data

    if data_args.streaming:
        raw_datasets["train"] = raw_datasets["train"].to_iterable_dataset(num_shards=8)
        raw_datasets["valid"] = raw_datasets["valid"].to_iterable_dataset(num_shards=8)
        
        world_size = torch.distributed.get_world_size()       
        if world_size > 1:
            rank = torch.distributed.get_rank()
            raw_datasets["train"] = split_dataset_by_node(raw_datasets["train"], rank=rank, world_size=world_size)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=training_args.seed, buffer_size=100000)
            raw_datasets["valid"] = split_dataset_by_node(raw_datasets["valid"], rank=rank, world_size=world_size)
    
    assert data_args.audio_column_name in next(iter(raw_datasets.values())).column_names, "missing audio or incorrect audio column name"
    assert data_args.text_column_name in next(iter(raw_datasets.values())).column_names, "missing text or incorrect text column name"

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    config.update({"suppress_tokens": model_args.suppress_tokens, "use_cache": False, "pif_loss_alpha": data_args.pif_loss_alpha})

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if peft_args.peft_type:

        model.setup_peft(peft_args)
        
        if peft_args.peft_type == "lora":
            from models.peft import mark_only_lora_as_trainable
            mark_only_lora_as_trainable(model)
            #assert model.model.encoder.weights_merged == False, "do not merge lora weights for training"

        if peft_args.peft_type == "adapter":
            from models.peft import mark_only_adapter_as_trainable
            mark_only_adapter_as_trainable(model)
        
        if peft_args.peft_type == "prompt_tuning":
            from models.peft import mark_only_prompt_as_trainable
            mark_only_prompt_as_trainable(model)

        if peft_args.peft_type == "prefix_tuning":
            from models.peft import mark_only_prefix_as_trainable
            mark_only_prefix_as_trainable(model)
        
        # adding peft for encoder only
        model.model.encoder.gradient_checkpointing = False
        model.model.decoder.gradient_checkpointing = False

    else:
        if model_args.freeze_encoder:
            model.freeze_encoder()
            model.model.encoder.gradient_checkpointing = False

        if model_args.freeze_decoder:
            model.freeze_decoder()
            model.model.decoder.gradient_checkpointing = False

    if data_args.language is not None and data_args.language != "en":
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)
        model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language=data_args.language, task=data_args.task)
        model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids

    # 6. Resample speech dataset if necessary
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )
    
    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]

    if data_args.use_pif:
        model_perturb_input = feature_extractor.model_input_names[1]

    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    ) or getattr(data_args, "use_pif", False)

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if data_args.max_eval_samples is not None:
        raw_datasets["valid"] = raw_datasets["valid"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
    
        sp_rate = batch["sp_rate"]
        vtlp_rate = batch["vtlp_rate"]
        pitch_level = batch["pitch_level"]
        
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask, 
            vtlp_rate=vtlp_rate, sp_rate=sp_rate, pitch_level=pitch_level, use_pif=data_args.use_pif,
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]

        if data_args.use_pif:
            batch[model_perturb_input] = inputs.get(model_perturb_input)[0]
            
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = tokenizer._normalize(batch[text_column_name])
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
        )
    
    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length
    
    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    
    # 8. Load Metric
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            if hasattr(config, "peft_config"):
                import dataclasses
                config.peft_config = dataclasses.asdict(config.peft_config)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)
    processor.feature_extractor = feature_extractor
    
    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
        use_pif=data_args.use_pif,
        pif_layer=data_args.pif_layer,
    )

    # 11. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["valid"],
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.patience)],
    )
    
    # 12. Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)  # Saves the feature extractor too for easy upload

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()

