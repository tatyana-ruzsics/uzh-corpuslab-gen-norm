#!/bin/bash
# Usage: ./Main-arch-sync.sh DATA_FOLDER_NAME RESULTS_FOLDER NMT_ENSEMBLES BEAM SYNC_MODEL_TYPE NMT_MODEL_TYPE NMT_MODEL_SPEC
# NMT ensemble evaluation
# ./Main-arch-sync.sh Archimob-pos arch 5 3 nmt
# NMT + HLLM
# ./Main-arch-sync.sh Archimob-pos arch 5 3 we norm_soft
# NMT+Gold POS ensemble evaluation
# ./Main-arch-sync.sh Archimob-pos arch 5 3 nmt norm_soft_pos
# NMT+Gold POS + HLLM
# ./Main-arch-sync.sh Archimob-pos arch 5 3 we norm_soft_pos
# NMT+Context/NMT+Context+Gold POS/NMT+Context+Pred POS ensemble evaluation
# ./Main-arch-sync.sh Archimob-pos arch 5 3 nmt norm_soft_char_context
# NMT+Context + HLLM
# ./Main-arch-sync.sh Archimob-pos arch 5 3 we norm_soft_char_context
# NMT+Context+Gold POS + HLLM
# ./Main-arch-sync.sh Archimob-pos arch 5 3 we norm_soft_char_context pos
# NMT+Context+Pred POS + HLLM
# ./Main-arch-sync.sh Archimob-pos arch 5 3 we norm_soft_char_context pos_aux

# DATA_FOLDER_NAME - where the data is saved
# Pretrained source context NMT model folders name pattern: {RESULTS_FOLDER}/{NMT_MODEL_TYPE}_{NMT_MODEL_SPEC}_{NMT_SEED} where NMT_SEED from 1 to NMT_ENSEMBLES
# is used to recontruct model folders from arguments

#Configuration options:
# w  - use lm over words(trained on the target data)
# we - use lm over words(trained on the target and extra target data)

###########################################
## POINTERS TO WORKING AND DATA DIRECTORIES
###########################################

export DIR=/home/tanja/uzh-corpuslab-gen-norm

# data paths
export DATA=$DIR/data/$1
if [[ $CONFIG == *"e"* ]]; then
export BIGLM=/gennorm/normalization/data/lm
fi
export TRAINDATA=$DATA/train.txt
export DEVDATA=$DATA/dev.txt
export TESTDATA=$DATA/test.txt

#LM paths
export LD_LIBRARY_PATH=/home/christof/Chintang/swig-srilm:$LD_LIBRARY_PATH
export PYTHONPATH=/home/christof/Chintang/swig-srilm:$PYTHONPATH
export PATH=/home/christof/Chintang/SRILM/bin:/home/christof/Chintang/SRILM/bin/i686-m64:$PATH

#MERT path
export MERT=/home/christof/Chintang/uzh-corpuslab-morphological-segmentation/zmert_v1.50

#Pretrained NMT model
export MODEL=$DIR/results/$2

export BEAM=$4

export CONFIG=$5

export NMT_TYPE=$6

if [ -z $7 ];
then
export NMT_FULL_TYPE=${NMT_TYPE}
else
export NMT_FULL_TYPE=${NMT_TYPE}_$7
fi

export INPUT_FORMAT="0,1,2"

export NMT_ENSEMBLES=$3

# results folder
mkdir -p $MODEL/${NMT_FULL_TYPE}_sync/ensemble/${CONFIG}
export RESULTS=$MODEL/${NMT_FULL_TYPE}_sync/ensemble/${CONFIG}

# pretrained models
nmt_predictors="nmt"
nmt_path="$MODEL/${NMT_FULL_TYPE}_1"
if [ $NMT_ENSEMBLES -gt 1 ]; then
while read num; do nmt_predictors+=",nmt"; done < <(seq $(($NMT_ENSEMBLES-1)))
while read num; do nmt_path+=",$MODEL/${NMT_FULL_TYPE}_$num"; done < <(seq 2 $NMT_ENSEMBLES)
fi
echo "$nmt_path"

###########################################
## PREPARATION - src/trg splits and vocabulary
###########################################

# Prepare train set
cut -f1 $TRAINDATA | grep . | tr '[:upper:]' '[:lower:]' > $RESULTS/train.src
cut -f2 $TRAINDATA | grep . | tr '[:upper:]' '[:lower:]' > $RESULTS/train.trg

# Prepare test set
cut -f1 $TESTDATA | grep . | tr '[:upper:]' '[:lower:]' > $RESULTS/test.src
cut -f2 $TESTDATA | grep . | tr '[:upper:]' '[:lower:]' > $RESULTS/test.trg

# Prepare validation set
cut -f1 $DEVDATA | grep . | tr '[:upper:]' '[:lower:]' > $RESULTS/dev.src
cut -f2 $DEVDATA | grep . | tr '[:upper:]' '[:lower:]' > $RESULTS/dev.trg

if [[ $CONFIG == *"e"* ]]; then
# Prepare training target file based on the extra data
cut -f2 $EXTRADATA | tr '[:upper:]' '[:lower:]'> $BIGLM/extra.train.trg
# Extend training set
cat $RESULTS/train.trg $BIGLM/extra.train.trg > $BIGLM/train_ext.trg
export EXTENDEDTRAIN=$BIGLM/train_ext.trg
fi



if [[ $CONFIG == "nmt" ]]; then # Only evaluate ensembles of NMT models

############################################
# NMT EVALUATION on test
############################################

    python $DIR/src/${NMT_TYPE}.py ensemble_test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=test.out $RESULTS --input_format=${INPUT_FORMAT}

    # evaluate on tokens - detailed output for the test set
    python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT}

    ##evaluate ambuguity on tokens - detailed output for the test set
    python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt  --input_format=${INPUT_FORMAT}

else # Source context NMT + HLLM

##########################################
# TRAIN HLLM over words
##########################################

# Use target extended data for language model over words
if [[ $CONFIG == *"e"* ]]; then

    # Build vocab over morphemes
    python $DIR/src/vocab_builder.py build $EXTENDEDTRAIN $BIGLM/morph_vocab.txt --segments
    # Apply vocab mapping
    python $DIR/src/vocab_builder.py apply $EXTENDEDTRAIN $BIGLM/morph_vocab.txt $BIGLM/train_ext.morph.itrg  --segments
    # train LM
    (ngram-count -text $BIGLM/train_ext.morph.itrg -lm $BIGLM/morfs.lm -order 3 -write $BIGLM/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $BIGLM/train_ext.morph.itrg -lm $BIGLM/morfs.lm -order 3 -write $BIGLM/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $BIGLM/train_ext.morph.itrg -lm $BIGLM/morfs.lm -order 3 -write $BIGLM/morfs.lm.counts -wbdiscount -interpolate );}

# if run many times - comment above and add the comment:
# echo "Using pretrained lm from $EXTENDEDTRAIN"

# Use only target train data for language model over words
else
    # Build vocab over morphemes
    python vocab_builder.py build $TRAINDATA $RESULTS/morph_vocab.txt --segments
    # Apply vocab mapping
    python vocab_builder.py apply $TRAINDATA $RESULTS/morph_vocab.txt $RESULTS/train.morph.itrg  --segments
    # train LM
    (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -wbdiscount -interpolate );}

fi


#################################################################
# MERT Optimization for NMT+HLLM decoding parameters + EVALUATION
#################################################################

mkdir $RESULTS/mert
export MERTEXPER=$RESULTS/mert

cd $MERTEXPER

if [[ $CONFIG == "w" ]]; then
# passed to zmert: commands to decode n-best list from dev file
echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/nbest.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$RESULTS/morfs.lm --output_format=1 --morph_vocab=$RESULTS/morph_vocab.txt --nmt_type=${NMT_TYPE} --input_format=${INPUT_FORMAT}"> SDecoder_cmd

# passed to zmert: commands to decode 1-best list from test file
echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$TESTDATA --pred_path=$MERTEXPER/test.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$RESULTS/morfs.lm --morph_vocab=$RESULTS/morph_vocab.txt --nmt_type=${NMT_TYPE} --input_format=${INPUT_FORMAT}" > SDecoder_cmd_test

echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\nnmt 1\nlm 0.1" > SDecoder_cfg.txt

#    echo -e "nmt\t|||\t1\tFix\t0\t+1\t0\t+1\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt
echo -e "nmt\t|||\t1\tOpt\t0\t+Inf\t0\t+1\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = absval 1 nmt" > params.txt

elif [[ $CONFIG == "we" ]]; then
    # passed to zmert: commands to decode n-best list from dev file
    echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/nbest.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$BIGLM/morfs.lm --output_format=1 --morph_vocab=$BIGLM/morph_vocab.txt --nmt_type=${NMT_TYPE} --input_format=${INPUT_FORMAT} --exclude_eow"> SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$TESTDATA --pred_path=$MERTEXPER/test.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$BIGLM/morfs.lm --morph_vocab=$BIGLM/morph_vocab.txt --nmt_type=${NMT_TYPE} --input_format=${INPUT_FORMAT} --exclude_eow" > SDecoder_cmd_test

    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\nnmt 1\nlm 0.1" > SDecoder_cfg.txt

#    echo -e "nmt\t|||\t1\tFix\t0\t+1\t0\t+1\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt
    echo -e "nmt\t|||\t1\tOpt\t0\t+Inf\t0\t+1\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = absval 1 nmt" > params.txt

else
 echo -e "Uknown configuration!"

fi

cp $DIR/src/ZMERT_cfg.txt $MERTEXPER
cp $RESULTS/dev.trg $MERTEXPER
cp $RESULTS/test.src $MERTEXPER

wait

java -cp $MERT/lib/zmert.jar ZMERT -maxMem 500 ZMERT_cfg.txt

## copy test out file - for analysis
cp test.out.predictions $RESULTS/test_out_mert.txt
cp test.out.eval $RESULTS/test.eval

## copy n-best file for dev set with optimal weights - for analysis
cp nbest.out.predictions $RESULTS/nbest_dev_mert.out
cp nbest.out.eval $RESULTS/dev.eval

cp SDecoder_cfg.txt.ZMERT.final $RESULTS/params-mert-ens.txt

##evaluate on tokens - detailed output for the test set
#if [[ $CONFIG == *"e"* ]]; then
#python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/test.eval.det $RESULTS/Errors_test.txt --extended_train_data=$EXTENDEDTRAIN  --input_format=${INPUT_FORMAT}
#else
python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/test.eval.det $RESULTS/Errors_test.txt  --input_format=${INPUT_FORMAT}
#fi

##evaluate ambuguity on tokens - detailed output for the test set
python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt   --input_format=${INPUT_FORMAT}
#rm -r $MERTEXPER

fi
