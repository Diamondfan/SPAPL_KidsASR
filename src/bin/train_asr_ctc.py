#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

import json
import logging
import os
import sys
import warnings
from dataclasses import field
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
from datasets import DatasetDict, IterableDatasetDict
from datasets.distributed import split_dataset_by_node

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

sys.path.append(os.environ['rootdir']+'/src')
from arguments import PEFTArguments
from ctc_arguments import ModelArguments, DataTrainingArguments
from data.whisper_loader import WhisperDataset
from data.data_utils import DataCollatorCTCWithPadding
from data.english_normalizer import EnglishTextNormalizer
from models.feature_extractor import Wav2Vec2FeatureExtractor, SeamlessM4TFeatureExtractor

feature_extractor_dict = {"wav2vec": Wav2Vec2FeatureExtractor, "hubert": Wav2Vec2FeatureExtractor, 
                         "wavlm": Wav2Vec2FeatureExtractor, "w2v2bert": SeamlessM4TFeatureExtractor}


logger = logging.getLogger(__name__)


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(dataset):
        all_text = []
        for i, sample in enumerate(dataset):
            all_text.append(sample["target_text"])
        
        all_text = " ".join(all_text)
        vocab = list(set(all_text))
        return vocab
    
    vocab_train = extract_all_chars(datasets["train"])
    vocab_valid = extract_all_chars(datasets["valid"])
    
    # take union of all unique characters in each dataset
    vocab_set = list(set(vocab_train) | set(vocab_valid))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PEFTArguments, TrainingArguments))
    if len(sys.argv) == 2: 
        if sys.argv[1].endswith(".json"):
            model_args, data_args, peft_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        elif sys.argv[1].endswith(".yaml"):
            model_args, data_args, peft_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
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
    
    # 2. normalize text
    
    text_column_name = data_args.text_column_name
    column_names = raw_datasets["train"].column_names
    
    if data_args.normalize_file:
        with open(data_args.normalize_file, encoding="utf-8") as vocab_handle:
            english_spelling_normalizer = json.load(vocab_handle)
        normalizer = EnglishTextNormalizer(english_spelling_normalizer)
        
        def remove_special_characters(batch):
            batch["target_text"] = normalizer(batch[text_column_name])
            return batch

        with training_args.main_process_first(desc="dataset map special characters removal"):
            raw_datasets = raw_datasets.map(
                remove_special_characters,
            )
        
        if english_spelling_normalizer is not None:
            normalizer_file_save = os.path.join(training_args.output_dir, "normalizer.json")
            with open(normalizer_file_save, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

    # save special tokens for tokenizer
    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    tokenizer_kwargs = {}
    if tokenizer_name_or_path is None:
        # save vocab in training output dir
        tokenizer_name_or_path = training_args.output_dir

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

        if os.path.exists(model_args.vocab_path):
            import shutil
            shutil.copyfile(model_args.vocab_path, vocab_file)
        
        with training_args.main_process_first(desc="dataset map vocabulary creation"):
            if not os.path.isfile(vocab_file):
                os.makedirs(tokenizer_name_or_path, exist_ok=True)
                vocab_dict = create_vocabulary_from_data(
                    raw_datasets,
                    word_delimiter_token=word_delimiter_token,
                    unk_token=unk_token,
                    pad_token=pad_token,
                )

                # save vocab dict to be loaded into tokenizer
                with open(vocab_file, "w") as file:
                    json.dump(vocab_dict, file)

        # if tokenizer has just been created
        # it is defined by `tokenizer_class` if present in config else by `model_type`
        tokenizer_kwargs = {
            "config": config if config.tokenizer_class is not None else None,
            "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
        }

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.
    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **tokenizer_kwargs,
    )
    feature_extractor_cls = feature_extractor_dict[data_args.feature_extractor_type]
    feature_extractor = feature_extractor_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "ctc_zero_infinity": model_args.ctc_zero_infinity,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
            "apply_spec_augment": model_args.apply_spec_augment,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
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
        # freeze encoder
        if model_args.freeze_feature_encoder:
            model.freeze_feature_encoder()
    
    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
    
    # make sure that dataset decodes audio with correct sampling rate
    dataset_sampling_rate = next(iter(raw_datasets["train"]))[data_args.audio_column_name]["sampling_rate"]
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # derive max & min input length for sample rate & max duration
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = data_args.phoneme_language

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        sp_rate = batch["sp_rate"]
        vtlp_rate = batch["vtlp_rate"]
        pitch_level = batch["pitch_level"]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"],
                                vtlp_rate=vtlp_rate, sp_rate=sp_rate, pitch_level=pitch_level,)
        
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        # take length of raw audio waveform
        batch["input_length"] = len(sample["array"].squeeze())

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=column_names,
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
        )

    # 7. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {metric: evaluate.load(metric, cache_dir=model_args.cache_dir) for metric in data_args.eval_metrics}

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    # Now save everything to be able to create a single processor later
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

    try:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    processor.feature_extractor = feature_extractor
    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, feature_extractor_input_name=feature_extractor_input_name
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["valid"],
        tokenizer=processor,
    )

    # 8. Finally, we can start training
    # Training
    checkpoint = None
    # use last checkpoint if exist
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
