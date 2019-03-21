#!/bin/bash
# Usage: ./Main-seg-sync.sh DATA_PREFIX DATA_NAME NMT_ENSEMBLES BEAM MODEL_TYPE SEG_DATA_SPLIT
# NMT ensemble evaluation
# ./Main-seg-sync.sh eng english 5 3 nmt 0
# NMT + HLLM
# ./Main-seg-sync.sh eng english 5 3 we 0

#Configuration options:
# w  - use lm over words(trained on the target data)
# we - use lm over words(trained on the target and extra target data)
# w2 - use two lm over words(one trained on the target and another one on the extra target data)


###########################################
## POINTERS TO WORKING AND DATA DIRECTORIES
###########################################

export PF=$1
export DIR=/home/tanja/uzh-corpuslab-gen-norm

# data paths
export DATA=$DIR/data/canonical-segmentation/$2
if [[ $CONFIG == *"e"* ]]; then
export EXTRADATA=/$DIR/data/canonical-segmentation/additional/${PF}/aspell.txt
fi

#LM paths
export LD_LIBRARY_PATH=/home/christof/Chintang/swig-srilm:$LD_LIBRARY_PATH
export PYTHONPATH=/home/christof/Chintang/swig-srilm:$PYTHONPATH
export PATH=/home/christof/Chintang/SRILM/bin:/home/christof/Chintang/SRILM/bin/i686-m64:$PATH

#MERT path
export MERT=/home/christof/Chintang/uzh-corpuslab-morphological-segmentation/zmert_v1.50

#Pretrained NMT model
export MODEL=/results/results/segm

export BEAM=$4

export CONFIG=$5

export n=$6

#for (( n=0; n<=9; n++ )) #data split (from 0 till 9)
#
#do
# Data paths depend on the data split
export TRAINDATA=$DATA/train${n}
export DEVDATA=$DATA/dev$n
export TESTDATA=$DATA/test$n

export INPUT_FORMAT="0,2,1"

export NMT_ENSEMBLES=$3

# results folder
mkdir -p $DIR/results/${PF}/ensemble/${CONFIG}_test_norm/$n
export RESULTS=$DIR/results/${PF}/ensemble/${CONFIG}_test_norm/$n

# pretrained models
nmt_predictors="nmt"
nmt_path="$MODEL/${PF}_${n}_nmt_1"
if [ $NMT_ENSEMBLES -gt 1 ]; then
while read num; do nmt_predictors+=",nmt"; done < <(seq $(($NMT_ENSEMBLES-1)))
while read num; do nmt_path+=",$MODEL/${PF}_${n}_nmt_$num"; done < <(seq 2 $NMT_ENSEMBLES)
fi
echo "$nmt_path"

###########################################
## PREPARATION - src/trg splits and vocabulary
###########################################

# Prepare train set
cut -f1 $TRAINDATA | tr '[:upper:]' '[:lower:]' > $RESULTS/train.src
cut -f3 $TRAINDATA | tr '[:upper:]' '[:lower:]' > $RESULTS/train.trg

# Prepare test set
cut -f1 $TESTDATA | tr '[:upper:]' '[:lower:]' > $RESULTS/test.src
cut -f3 $TESTDATA | tr '[:upper:]' '[:lower:]' > $RESULTS/test.trg

# Prepare validation set
cut -f1 $DEVDATA | tr '[:upper:]' '[:lower:]' > $RESULTS/dev.src
cut -f3 $DEVDATA | tr '[:upper:]' '[:lower:]' > $RESULTS/dev.trg

if [[ $CONFIG == *"e"* ]]; then
# Prepare training target file based on the extra data
cut -f1 $EXTRADATA | tr '[:upper:]' '[:lower:]' > $RESULTS/extra.train.trg
# Extend training set
cat $RESULTS/train.trg $RESULTS/extra.train.trg > $RESULTS/train_ext.trg
fi



if [[ $CONFIG == "nmt" ]]; then # Only evaluate ensembles of NMT models

############################################
# NMT EVALUATION on test
############################################

    python $DIR/src/norm_soft.py ensemble_test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=test.out $RESULTS --input_format=${INPUT_FORMAT}

    # evaluate on tokens - detailed output for the test set
    python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT}

else # NMT + LM

##########################################
# Train HLLLM over segments
##########################################

# Use target extended data for language model over segments
if [[ $CONFIG == "we" ]]; then

    # Build vocab over morphemes
    python vocab_builder.py build $RESULTS/train_ext.trg $RESULTS/morph_vocab.txt --segments
    # Apply vocab mapping
    python vocab_builder.py apply $RESULTS/train_ext.trg $RESULTS/morph_vocab.txt $RESULTS/train_ext.morph.itrg --segments
    # train LM
    (ngram-count -text $RESULTS/train_ext.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train_ext.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train_ext.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -wbdiscount -interpolate );}

# Use only target train data for language model over segments
elif [[ $CONFIG == "w" ]]; then
    # Build vocab over morphemes
    python vocab_builder.py build $RESULTS/train.trg $RESULTS/morph_vocab.txt --segments
    # Apply vocab mapping
    python vocab_builder.py apply $RESULTS/train.trg $RESULTS/morph_vocab.txt $RESULTS/train.morph.itrg --segments
    # train LM
    (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -wbdiscount -interpolate );}

# Use only target train data for lm 1, extended data for lm2
elif [[ $CONFIG == "w2" ]]; then
    # Build vocab over morphemes over train + extended data
    python vocab_builder.py build $RESULTS/train_ext.trg $RESULTS/morph_vocab.txt --segments
    # Apply vocab mapping
    python vocab_builder.py apply $RESULTS/train.trg $RESULTS/morph_vocab.txt $RESULTS/train.morph1.itrg --segments
    # train LM1
    (ngram-count -text $RESULTS/train.morph1.itrg -lm $RESULTS/morfs.lm1 -order 3 -write $RESULTS/morfs.lm1.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train.morph1.itrg -lm $RESULTS/morfs.lm1 -order 3 -write $RESULTS/morfs.lm1.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train.morph1.itrg -lm $RESULTS/morfs.lm1 -order 3 -write $RESULTS/morfs.lm1.counts -wbdiscount -interpolate );}
    # Apply vocab mapping
    PYTHONIOENCODING=utf8  python vocab_builder.py apply $RESULTS/extra.train.trg $RESULTS/morph_vocab.txt $RESULTS/train.morph2.itrg --segments
    # train LM2
    (ngram-count -text $RESULTS/train.morph2.itrg -lm $RESULTS/morfs.lm2 -order 3 -write $RESULTS/morfs.lm2.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train.morph2.itrg -lm $RESULTS/morfs.lm2 -order 3 -write $RESULTS/morfs.lm2.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train.morph2.itrg -lm $RESULTS/morfs.lm2 -order 3 -write $RESULTS/morfs.lm2.counts -wbdiscount -interpolate );}

fi


#################################################################
# MERT Optimization for NMT+HLLM decoding parameters + EVALUATION
#################################################################

mkdir $RESULTS/mert
export MERTEXPER=$RESULTS/mert

cd $MERTEXPER

# NMT + two Language Model over segments
if [[ $CONFIG == "w2" ]]; then
    # passed to zmert: commands to decode n-best list from dev file
    echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/nbest.out --lm_predictors=srilm_morph,srilm_morph --lm_orders=3,3 --lm_paths=$RESULTS/morfs.lm1,$RESULTS/morfs.lm2 --output_format=1 --input_format=${INPUT_FORMAT} --morph_vocab=$RESULTS/morph_vocab.txt"> SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$TESTDATA --pred_path=$MERTEXPER/test.out --lm_predictors=srilm_morph,srilm_morph --lm_orders=3,3 --lm_paths=$RESULTS/morfs.lm1,$RESULTS/morfs.lm2 --input_format=${INPUT_FORMAT} --morph_vocab=$RESULTS/morph_vocab.txt" > SDecoder_cmd_test

# lm1: over train trg, lm2: over extra trg
    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\nnmt 1\nlm1 0.1\nlm2 0.1" > SDecoder_cfg.txt

    echo -e "nmt\t|||\t1\tFix\t0\t+1\t0\t+1\nlm1\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nlm2\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt

# NMT + Language Model over segments
elif [[ $CONFIG == "w" ]] || [[ $CONFIG == "we" ]]; then
    # passed to zmert: commands to decode n-best list from dev file
    echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/nbest.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$RESULTS/morfs.lm --output_format=1 --input_format=${INPUT_FORMAT} --morph_vocab=$RESULTS/morph_vocab.txt"> SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$TESTDATA --pred_path=$MERTEXPER/test.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$RESULTS/morfs.lm --input_format=${INPUT_FORMAT} --morph_vocab=$RESULTS/morph_vocab.txt" > SDecoder_cmd_test

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
#if [[ $CONFIG == "w2" ]] || [[ $CONFIG == "we" ]]; then
#python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT}  --extended_train_data=$RESULTS/extra.train.trg
#else
python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT}
#fi
#rm -r $MERTEXPER

fi

echo "Process {$n} finished"
#)

#done
