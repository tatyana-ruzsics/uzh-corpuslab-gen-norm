#!/bin/bash
# Usage: ./Main-lemm-eval.sh TRAIN_DEV_DATA_FOLDER_NAME TEST_DEV_DATA_FOLDER_NAME RESULTS_FOLDER MODEL_TYPE ENS/INDIVIDUAL NMT_ENSEMBLES/NMT_SEED BEAM NMT_MODEL_SPEC
# Baseline evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque baseline
# NMT individual model evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft ind 1 3
# NMT ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft ens 5 3
# NMT+Gold POS ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_pos ens 5 3
# NMT+Context ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_char_context ens 5 3
# NMT+Context+Gold POS ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_char_context ens 5 3 pos
# NMT+Context+Pred POS ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_char_context ens 5 3 pos_aux
# NMT+Context+Pred POS with no lowercasing ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_char_context ens 5 3 pos_aux_no_low
# NMT+Pred POS ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_char_context ens 5 3 pos_aux_no_cont
# NMT+Pred POS with no lowercasing ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_char_context ens 5 3 pos_aux_no_cont_no_low
# NMT+Pred POS with mask cap ensemble evaluation
# Usage: ./Main-lemm-eval.sh ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque norm_soft_char_context ens 5 3 pos_aux_no_cont_cap_mask


# *_DATA_FOLDER_NAME - where the data is saved
# Pretrained source context NMT model folders name pattern: {RESULTS_FOLDER}/{NMT_MODEL_TYPE}_{NMT_MODEL_SPEC}_{NMT_SEED}
# where NMT_SEED from 1 to NMT_ENSEMBLES
# is used to recontruct model folders from arguments

###########################################
## POINTERS TO WORKING AND DATA DIRECTORIES
###########################################
#

export DIR=/home/tanja/uzh-corpuslab-gen-norm

#export DATADIR=/gennorm/lemmatization

export INPUT_FORMAT="1,2,3"

####BASELINE####
if [[ $4 == 'baseline' ]]; then

# data paths
# export TRAINDATA=$DATADIR/data/$1-ud-train.conllu
# export DEVDATA=$DATADIR/data/$1-ud-dev.conllu
# export TESTDATA=$DATADIR/data/$2.conllu
export TRAINDATA=/$1-ud-train.conllu
export DEVDATA=/$1-ud-dev.conllu
export TESTDATA=/$2.conllu

# results folder
#export MODEL=$DATADIR/results/$3
export MODEL=$3
mkdir -p $MODEL/baseline
export RESULTS=$MODEL/baseline

# eval baseline
python $DIR/src/accuracy-det.py eval_baseline $TRAINDATA $DEVDATA  --error_file=$RESULTS/Errors_baseline_dev.txt --input_format=${INPUT_FORMAT} --conll_format > $RESULTS/baseline.dev.eval
python $DIR/src/accuracy-det.py eval_baseline $TRAINDATA $TESTDATA  --error_file=$RESULTS/Errors_baseline_test.txt --input_format=${INPUT_FORMAT} --conll_format > $RESULTS/baseline.test.eval

# eval baseline - pos
python $DIR/src/accuracy-det.py eval_ambiguity_baseline $TRAINDATA $DEVDATA  --error_file=$RESULTS/Errors_baseline_pos_dev.txt --input_format=${INPUT_FORMAT} --conll_format > $RESULTS/baseline-pos.dev.eval
python $DIR/src/accuracy-det.py eval_ambiguity_baseline $TRAINDATA $TESTDATA  --error_file=$RESULTS/Errors_baseline_pos_test.txt --input_format=${INPUT_FORMAT} --conll_format > $RESULTS/baseline-pos.test.eval

else
####NMT####

# data paths
# export TRAINDATA=$DATADIR/data/$1-ud-train.conllu
# export DEVDATA=$DATADIR/data/$1-ud-dev.conllu
# export TESTDATA=$DATADIR/data/$2.conllu
export TRAINDATA=/$1-ud-train.conllu
export DEVDATA=/$1-ud-dev.conllu
export TESTDATA=/$2.conllu

#Pretrained NMT model
#export MODEL=$DATADIR/results/$3
#export MODEL=$DATADIR/$3
export MODEL=$3
export BEAM=$7
export NMT_TYPE=$4

if [ -z $8 ];
then
export NMT_FULL_TYPE=${NMT_TYPE}
else
export NMT_FULL_TYPE=${NMT_TYPE}_$8
fi

## ensemble model
if [[ $5 == 'ens' ]]; then

export NMT_ENSEMBLES=$6

# results folder
mkdir -p $MODEL/${NMT_FULL_TYPE}_ens5
export RESULTS=$MODEL/${NMT_FULL_TYPE}_ens5

# pretrained models
nmt_predictors="nmt"
nmt_path="$MODEL/${NMT_FULL_TYPE}_1"
if [ $NMT_ENSEMBLES -gt 1 ]; then
while read num; do nmt_predictors+=",nmt"; done < <(seq $(($NMT_ENSEMBLES-1)))
while read num; do nmt_path+=",$MODEL/${NMT_FULL_TYPE}_$num"; done < <(seq 2 $NMT_ENSEMBLES)
fi
echo "$nmt_path"

# ensemble predictions
if [[ $8 == *"no_low"* ]]; then
python $DIR/src/${NMT_TYPE}.py ensemble_test ${nmt_path} --test_path=$DEVDATA --beam=$BEAM --pred_path=dev.out $RESULTS --input_format=${INPUT_FORMAT} --conll_format --lowercase=False
python $DIR/src/${NMT_TYPE}.py ensemble_test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=test.out $RESULTS --input_format=${INPUT_FORMAT} --conll_format --lowercase=False
else
python $DIR/src/${NMT_TYPE}.py ensemble_test ${nmt_path} --test_path=$DEVDATA --beam=$BEAM --pred_path=dev.out $RESULTS --input_format=${INPUT_FORMAT} --conll_format
python $DIR/src/${NMT_TYPE}.py ensemble_test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=test.out $RESULTS --input_format=${INPUT_FORMAT} --conll_format
fi	

#
# evaluate ensemble on tokens - detailed output
if  [[ $8 == *'pos_aux'* ]]; then
	if [[ $8 == *"no_low"* ]]; then
		python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
		python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
	else
		python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
		python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
	fi
else
python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt --input_format=${INPUT_FORMAT} --conll_format
python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT} --conll_format
fi

##evaluate ensemble - ambuguity on tokens
if  [[ $8 == *'pos_aux'* ]]; then
	if [[ $8 == *"no_low"* ]]; then
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
	else
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
	fi
else
python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/dev.out.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt --input_format=${INPUT_FORMAT} --conll_format
python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt --input_format=${INPUT_FORMAT} --conll_format
fi	

else
## individual model
export NMT_SEED=$6
export RESULTS=$MODEL/${NMT_FULL_TYPE}_${NMT_SEED}

if [[ $8 == *"no_low"* ]]; then
python $DIR/src/${NMT_TYPE}.py test ${nmt_path} --test_path=$DEVDATA --beam=$BEAM --pred_path=best.dev.3 $RESULTS --input_format=${INPUT_FORMAT} --conll_format --lowercase=False
python $DIR/src/${NMT_TYPE}.py test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=best.test.3 $RESULTS --input_format=${INPUT_FORMAT} --conll_format --lowercase=False
else
python $DIR/src/${NMT_TYPE}.py test ${nmt_path} --test_path=$DEVDATA --beam=$BEAM --pred_path=best.dev.3 $RESULTS --input_format=${INPUT_FORMAT} --conll_format
python $DIR/src/${NMT_TYPE}.py test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=best.test.3 $RESULTS --input_format=${INPUT_FORMAT} --conll_format
fi


## evaluate - detailed output
if  [[ $8 == *'pos_aux'* ]]; then
	if [[ $8 == *"no_low"* ]]; then
		python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/best.dev.3.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
		python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/best.test.3.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
	else
		python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/best.dev.3.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
		python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/best.test.3.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
	fi
else
python $DIR/src/accuracy-det.py eval $TRAINDATA $DEVDATA $RESULTS/best.dev.3.predictions $RESULTS/dev.eval.det $RESULTS/Errors_dev.txt --input_format=${INPUT_FORMAT} --conll_format
python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/best.test.3.predictions $RESULTS/test.eval.det $RESULTS/Errors_test.txt --input_format=${INPUT_FORMAT} --conll_format
fi


##evaluate - ambuguity on tokens - detailed output for the test set
if  [[ $8 == *'pos_aux'* ]]; then
	if [[ $8 == *"no_low"* ]]; then
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/best.dev.3.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/best.test.3.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics --lowercase=False
	else
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/best.dev.3.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
		python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/best.test.3.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt --input_format=${INPUT_FORMAT} --conll_format --pos_statistics
	
	fi
else
python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $DEVDATA $RESULTS/best.dev.3.predictions $RESULTS/dev.eval.det.pos $RESULTS/Errors_dev_pos.txt --input_format=${INPUT_FORMAT} --conll_format
python $DIR/src/accuracy-det.py eval_ambiguity $TRAINDATA $TESTDATA $RESULTS/best.test.3.predictions $RESULTS/test.eval.det.pos $RESULTS/Errors_test_pos.txt --input_format=${INPUT_FORMAT} --conll_format

fi

fi
fi
