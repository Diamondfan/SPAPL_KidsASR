#!/usr/bin/env bash

# 2023-2024 (Ruchao Fan SPAPL)

export rootdir=/data/ruchao/workdir/SPAPL_KidsASR/
data=/data/Databases/MyST/myst-v0.4.2/

stage=5
end_stage=5

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # format the data as Kaldi data directories and prepare the raw data
  # e.g. data/train_raw with transcription and data/train_raw_wotrn without transcription

  for x in train development test; do
    local/myst_data_prepare.sh $data/data/$x data/${x}_raw
  done

  echo "[Stage 1] Raw Data Preparation Finished."
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  # whisper-based filtering, using whisper large-v2 model to filter those utterance with WER > 50%

  WER_Threshold=0.5
  filter_n_words=3
  remove_long_duration=0
  model_name="openai/whisper-large-v2" #

  for x in train_raw development_raw test_raw; do

    if [ $x == "test_raw" ]; then
      remove_long_duration=0
    else
      remove_long_duration=30
    fi
    
    CUDA_VISIBLE_DEVICES="0" python local/whisper_filter.py \
      --wav_scp data/$x/wav.scp \
      --trn_scp data/$x/text \
      --model $model_name \
      --wer_threshold $WER_Threshold \
      --remove_n_words $filter_n_words \
      --remove_long_dur $remove_long_duration \
      --saved_utt_list data/$x/whisper_filter_list

  done

  echo "[stage 2] Data Pre-processing, whisper-based filitering bad quality utterance finished!"
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then

  for x in train development test; do 
    utils/subset_data_dir.sh --utt-list data/${x}_raw/whisper_filter_list data/${x}_raw data/${x}_filter
  done
  echo "[Stage 3] Get the filtered data dir!"
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then

  for x in train_raw train_filter development_raw development_filter test_raw test_filter; do 
    python local/compute_duration.py --wav_scp data/$x/wav.scp --trn_scp data/$x/text 
  done
  echo "[Stage 4] Compute duration of each dataset Finished!"
fi

if [ $stage -le 5 ] && [ $end_stage -ge 5 ]; then

  for x in test_filter; do 
    python local/split_datasets_duration.py \
      --raw_dataset data/$x/ \
      --dur_threshold 30 \
      --short_list data/$x/lt30s.utt \
      --long_list data/$x/gt30s.utt 

    ./utils/subset_data_dir.sh --utt-list data/$x/lt30s.utt data/$x/ data/${x}_lt30/
    ./utils/subset_data_dir.sh --utt-list data/$x/gt30s.utt data/$x/ data/${x}_gt30/
  done
  echo "[Stage 5] Split datasets according to duration"
fi




