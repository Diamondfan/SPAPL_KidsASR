#!/usr/bin/env bash

# 2023-2024 (Ruchao Fan)
# experiments for whisper model

export rootdir=/data/ruchao/workdir/SPAPL_KidsASR/
export PATH=$PATH:/data/ruchao/workdir/kaldi/tools/sctk/bin/:$rootdir/src/bin:

stage=2
end_stage=2

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # decode myst development and test sets with openai whisper models
  # models: 
  # openai/whisper-tiny.en     39M
  # openai/whisper-base.en     74M
  # openai/whisper-small.en    244M
  # openai/whisper-medium.en   769M
  # openai/whisper-large       1550M
  # openai/whisper-large-v2    1550M

  models="tiny.en base.en small.en medium.en large large-v2"
  comupte_wer=true     # in python code
  using_sclite=true    # post python code
  chunk_length=30
  expdir=exp/whisper_zero_shot

  for model in $models; do
    model_name=openai/whisper-$model
    echo "Evaluating Model: $model_name"

    for x in test_filter_lt30 test_filter_gt30 development_filter; do # test_raw development_raw; do # test_filter_lt30 

      resultdir=$expdir/$model/${x}/
      [ ! -d $resultdir ] && mkdir -p $resultdir

      CUDA_VISIBLE_DEVICES="0" decode_asr.py \
        --wav_scp data/$x/wav.scp \
        --trn_scp data/$x/text \
        --model $model_name \
        --processor $model_name \
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
  
  echo "[Stage 1] Evaluation of Whisper Models Finished."
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  # Finetuning Whisper Model

  #exp_dir="exp/whisper_small_en_trans_fullfinetuning_lr1e-5_2gpus_specaug_splibrosa_4ksteps/"
  #exp_dir="exp/whisper_small_en_trans_pifinttuning0p05_cnnout_lr1e-5_2gpus_vtlp0.9_1.1_4ksteps"
  exp_dir="exp/whisper_small_en_trans_encft_lr1e-5_2gpus_4ksteps"
  #exp_dir="exp/whisper_small_en_trans_lora_dec_dim8_alpha128_lr1e-3_2gpus_4ksteps/"
  #exp_dir="exp/whisper_small_finetuning_adapter_allenc_lr1e-4_bn64_zeroinit_2gpus_4ksteps/"

  [ ! -d $exp_dir ] && mkdir -p $exp_dir

  train_config=conf/whisper_small_train.yaml

  CUDA_VISIBLE_DEVICES="2,3" torchrun --rdzv-endpoint=localhost:21221 \
 	  --nproc_per_node 2 $rootdir/src/bin/train_asr.py $train_config > $exp_dir/train.log 2>&1 &
  
  echo "[Stage 2] Finetuning Whisper Models Finished."
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  # Evaluation of the finetuned Whisper Model

  #exp_dir="exp/whisper_small_en_trans_fullfinetuning_lr1e-4_2gpus_4ksteps"
  #exp_dir="exp/whisper_small_en_trans_fullfinetuning_lr1e-5_2gpus_specaug_vtlp0.9_1.1_4ksteps/"
  exp_dir="exp/whisper_small_en_trans_fullfinetuning_lr1e-5_2gpus_specaug_pitchpert2_4ksteps/"
  #exp_dir="exp/whisper_small_en_trans_fullfinetuning_lr1e-5_2gpus_specaug_splibrosa_4ksteps/"
  #exp_dir="exp/whisper_small_en_trans_decft_lr1e-5_2gpus_4ksteps/"
  #exp_dir="exp/whisper_small_en_trans_lora_enc_dim8_alpha128_lr1e-3_2gpus_4ksteps/"
  #exp_dir="exp/whisper_small_finetuning_adapter_allenc_lr1e-4_bn64_zeroinit_2gpus_4ksteps/"

  comupte_wer=true     # in python code
  using_sclite=true    # post python code
  chunk_length=30

  for x in development_filter test_filter_lt30 test_filter_gt30; do #; do #development_filter test_filter; do #

    checkpoints="checkpoint-4000"
    for checkpoint in $checkpoints; do
      resultdir=$exp_dir/$checkpoint/${x}/
      [ ! -d $resultdir ] && mkdir -p $resultdir

      CUDA_VISIBLE_DEVICES="2" decode_asr.py \
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