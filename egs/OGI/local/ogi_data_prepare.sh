
wav_dir=$1
data=$2

text=$wav_dir/docs/all.map
tmpdir=./tmp
mkdir -p $tmpdir

find $wav_dir/speech/scripted/ -iname "*.wav" > $tmpdir/all.flist

for x in train dev test; do
  [ ! -d $data/$x ] & mkdir -p $data/$x
  grep -f conf/$x.list $tmpdir/all.flist > $tmpdir/$x.wav

  sed -e 's:.*/\(.*\)/\(.*\).wav$:\2 \1:' $tmpdir/$x.wav | sort -k1,1 > $data/$x/utt2spk
  sed -e 's:.*/\(.*\)/.*/.*/\(.*\).wav$:\2 \1:' $tmpdir/$x.wav | sort -k1,1 > $data/$x/utt2age
  sed -e 's:.*/\(.*\).wav$:\1:' $tmpdir/$x.wav > $data/$x/all.uttids
  paste -d' ' $data/$x/all.uttids $tmpdir/$x.wav | sort -k1,1 > $data/$x/wav.scp
  python local/genText.py $text $data/$x/all.uttids $data/$x/text.unsort
  cat $data/$x/text.unsort | sort -k1,1 > $data/$x/text
  rm -f $data/$x/all.uttids $data/$x/text.unsort

  spk2utt=$data/$x/spk2utt
  utils/utt2spk_to_spk2utt.pl < $data/$x/utt2spk >$spk2utt || exit 1

  age2utt=$data/$x/age2utt
  utils/utt2spk_to_spk2utt.pl < $data/$x/utt2age >$age2utt || exit 1

  ntrans=$(wc -l <$data/$x/text)
  nutt2spk=$(wc -l <$data/$x/utt2spk)
  ! [ "$ntrans" -eq "$nutt2spk" ] && \
    echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1;

  utils/validate_data_dir.sh --no-feats $data/$x || exit 1;
done
rm -r $tmpdir

echo "$0: successfully prepared data in $data"
exit 0
