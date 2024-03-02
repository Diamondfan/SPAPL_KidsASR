#!/usr/bin/env bash

# 2023-2024 (Ruchao Fan)
# experiments for CTC model

export rootdir=/data/ruchao/workdir/SPAPL_KidsASR/
export PATH=$PATH:/data/ruchao/workdir/kaldi/tools/sctk/bin/:$rootdir/src/bin:

stage=1
end_stage=1

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  # Finetuning HuBERT WAVLM model with CTC loss
  #exp_dir="exp/wavlm_large_fullfinetuning_lr3e-4_2gpus_bth4_grad4_8ksteps/"
  #exp_dir="exp/hubert_large_fullfinetuning_lr3e-4_2gpus_bth4_grad4_8ksteps/"
  exp_dir="exp/wav2vec2_large_fullfinetuning_lr3e-4_2gpus_bth4_grad4_8ksteps/"

  [ ! -d $exp_dir ] && mkdir -p $exp_dir

  train_config=conf/hubert_train.yaml
  #train_config=conf/wavlm_train.yaml

  CUDA_VISIBLE_DEVICES="2" torchrun --rdzv-endpoint=localhost:12122 \
 	  --nproc_per_node 1 $rootdir/src/bin/train_asr_ctc.py $train_config #> $exp_dir/train.log 2>&1 &
  
  echo "[Stage 1] Finetuning CTC Models Finished."
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  # Evaluation of the Finetuned CTC model

  #exp_dir="exp/wavlm_large_fullfinetuning_lr3e-4_2gpus_bth4_grad4_8ksteps/"
  #exp_dir="exp/hubert_large_fullfinetuning_lr3e-4_2gpus_bth4_grad4_8ksteps/"
  exp_dir="exp/wav2vec2_large_fullfinetuning_lr3e-4_2gpus_bth4_grad4_8ksteps/"

  comupte_wer=true     # in python code
  using_sclite=true    # post python code
  chunk_length=30

  for x in dev test spont_all; do

    checkpoints="checkpoint-8000"
    for checkpoint in $checkpoints; do
      resultdir=$exp_dir/$checkpoint/${x}/
      [ ! -d $resultdir ] && mkdir -p $resultdir

      CUDA_VISIBLE_DEVICES="2" decode_asr_ctc.py \
        --wav_scp data/$x/wav.scp \
        --trn_scp data/$x/text \
        --model $exp_dir/$checkpoint \
        --processor $exp_dir \
        --compute_wer $comupte_wer \
        --result_ref_file $resultdir/ref.txt \
        --result_hyp_file $resultdir/hyp.txt \
        --chunk_length $chunk_length > $resultdir/decode.log 2>&1
        
      if [ $using_sclite ]; then
        echo "compute WER using sclite for $x"
        sclite -r $resultdir/ref.txt -h $resultdir/hyp.txt -i rm -o all stdout > $resultdir/result.wrd.txt
      fi
    done
  done
fi