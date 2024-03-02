
wav_dir=$1
data=$2
test_ratio=$3

text=$wav_dir/docs/all.map
tmpdir=./tmp
mkdir -p $tmpdir

find $wav_dir/speech/scripted/ -iname "*.wav" > $tmpdir/all.flist

[ ! -d $data/local/data ] & mkdir -p $data/local/data
newdata=$data/local/data

sed -e 's:.*/\(.*\)/\(.*\).wav$:\2 \1:' $tmpdir/all.flist | sort -k1,1 > $newdata/utt2spk
sed -e 's:.*/\(.*\)/.*/.*/\(.*\).wav$:\2 \1:' $tmpdir/all.flist | sort -k1,1 > $newdata/utt2age
sed -e 's:.*/\(.*\).wav$:\1:' $tmpdir/all.flist > $newdata/all.uttids
paste -d' ' $newdata/all.uttids $tmpdir/all.flist | sort -k1,1 > $newdata/wav.scp
python local/genText.py $text $newdata/all.uttids $newdata/text.unsort
cat $newdata/text.unsort | sort -k1,1 > $newdata/text

spk2utt=$newdata/spk2utt
utils/utt2spk_to_spk2utt.pl < $newdata/utt2spk >$spk2utt || exit 1

age2utt=$newdata/age2utt
utils/utt2spk_to_spk2utt.pl < $newdata/utt2age >$age2utt || exit 1

ntrans=$(wc -l <$newdata/text)
nutt2spk=$(wc -l <$newdata/utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1;

utils/validate_data_dir.sh --no-feats $newdata || exit 1;
rm -r $tmpdir

utils/subset_data_dir_tr_cv.sh --cv-spk-percent $test_ratio $newdata \
    $data/train $data/local/test_dev

utils/subset_data_dir_tr_cv.sh --cv-spk-percent 25 $data/local/test_dev \
    $data/test $data/dev

cut -d" " -f1 $data/train/utt2spk > conf/train.list
cut -d" " -f1 $data/test/utt2spk > conf/test.list
cut -d" " -f1 $data/dev/utt2spk > conf/dev.list

echo "$0: successfully prepared data in $data"
exit 0
