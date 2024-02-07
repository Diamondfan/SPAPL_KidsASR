
exp=exp/whisper_zero_shot/
search='tiny'  # check the specific decoding results
word='wrd'

. utils/parse_options.sh

for testset in development test; do

  for x in $exp/$search*/$testset*; do
    
    echo $x

    grep Sum/Avg $x/result.$word.txt
  
  done

done
