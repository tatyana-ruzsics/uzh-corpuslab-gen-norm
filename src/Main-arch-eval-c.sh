#!/bin/bash
# Usage: ./Main-arch-eval.sh DATA_FOLDER_NAME RESULTS_FOLDER NMT_ENSEMBLES BEAM SYNC_MODEL_TYPE NMT_MODEL_TYPE NMT_MODEL_SPEC
# NMT ensemble evaluation
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 nmt
# NMT + HLLM
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 we norm_soft


# NMT+Gold POS ensemble evaluation
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 nmt norm_soft_pos
# NMT+Context/NMT+Context+Gold POS/NMT+Context+Pred POS ensemble evaluation
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 nmt norm_soft_char_context
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 nmt norm_soft_char_context pos
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 nmt norm_soft_char_context pos_aux


# NMT+Gold POS + HLLM
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 we norm_soft_pos
# NMT+Context + HLLM
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 we norm_soft_char_context
# NMT+Context+Gold POS + HLLM
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 we norm_soft_char_context pos
# NMT+Context+Pred POS + HLLM
# ./Main-arch-eval.sh Archimob-pos arch-8-context-shuff 5 3 we norm_soft_char_context pos_aux

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

export CONFIG=$5

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
export MODEL=/gennorm/normalization/results/$2

export BEAM=$4



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



if [[ $CONFIG == "nmt" ]]; then # Only evaluate ensembles of NMT models

############################################
# NMT EVALUATION on dev
############################################
    
    python $DIR/src/${NMT_TYPE}.py ensemble_test ${nmt_path} --test_path=$DEVDATA --beam=$BEAM --pred_path=dev.out $RESULTS --input_format=${INPUT_FORMAT}

    # evaluate on tokens - detailed output for the dev set
    python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt --input_format=${INPUT_FORMAT}

    ##evaluate ambuguity on tokens - detailed output for the dev set
    python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt  --input_format=${INPUT_FORMAT}

############################################
# NMT EVALUATION on test
############################################

    # python $DIR/src/${NMT_TYPE}.py ensemble_test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=test.out $RESULTS --input_format=${INPUT_FORMAT}

    # # evaluate on tokens - detailed output for the test set
    # python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT}

    # ##evaluate ambuguity on tokens - detailed output for the test set
    # python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt  --input_format=${INPUT_FORMAT}

else # Source context NMT + HLLM

############################################
# NMT EVALUATION on dev
############################################

export w_nmt=$(awk '/nmt/ {print $2}' "$RESULTS/params-mert-ens.txt")
export w_lm=$(awk '/lm/ {print $2}' "$RESULTS/params-mert-ens.txt")
echo "python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/dev.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$BIGLM/morfs.lm --morph_vocab=$BIGLM/morph_vocab.txt --nmt_type=${NMT_TYPE} --input_format=${INPUT_FORMAT} --exclude_eow --predictor_weights ${w_nmt},${w_lm}"
# python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=dev.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$BIGLM/morfs.lm --morph_vocab=$BIGLM/morph_vocab.txt --nmt_type=${NMT_TYPE} --input_format=${INPUT_FORMAT} --exclude_eow --predictor_weights ${w_nmt},${w_lm}

##evaluate on tokens - detailed output for the dev set
python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt  --input_format=${INPUT_FORMAT}

##evaluate ambuguity on tokens - detailed output for the dev set
python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt   --input_format=${INPUT_FORMAT}



############################################
# NMT EVALUATION on test
############################################
##evaluate on tokens - detailed output for the test set
# python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/test.eval.det $RESULTS/Errors_test.txt  --input_format=${INPUT_FORMAT}

# ##evaluate ambuguity on tokens - detailed output for the test set
# python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt   --input_format=${INPUT_FORMAT}


fi
