#!/usr/bin/env bash

# 2023-2024 (Ruchao Fan SPAPL)

export rootdir=/data/ruchao/workdir/SPAPL_KidsASR/
data=/data/Databases/OGI_Kids/

stage=2
end_stage=2

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # format the data as Kaldi data directories and prepare the raw data
  # e.g. data/train_raw with transcription and data/train_raw_wotrn without transcription

  local/ogi_data_prepare.sh $data/ data/
  local/ogi_spon_data_prepare.sh $data data/
  
  echo "[Stage 1] Raw Data Preparation Finished."
fi


if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then

  for x in train dev test spont_all; do 
    python local/compute_duration.py --wav_scp data/$x/wav.scp --trn_scp data/$x/text --write_utt2dur data/$x/utt2dur
  done
  echo "[Stage 2] Compute duration of each dataset Finished!"
fi





