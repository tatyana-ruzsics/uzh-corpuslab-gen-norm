#!/bin/bash
# Usage Main-lemm-train.sh model_file_name data_folder_train_dev data_folder_test results_folder_name seed use_aux_loss_and/or_pos_feat
# NMT
# ./Main-lemm-train.sh norm_soft ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1
# NMT+Gold POS
# ./Main-lemm-train.sh norm_soft_pos ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1
# NMT+Context
# ./Main-lemm-train.sh norm_soft_char_context ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1
# NMT+Context+Gold POS
# ./Main-lemm-train.sh norm_soft_char_context ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1 pos
# NMT+Context+Pred POS
# ./Main-lemm-train.sh norm_soft_char_context ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1 pos_aux
# NMT+Context+Pred POS with no lowercasing
# ./Main-lemm-train.sh norm_soft_char_context ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1 pos_aux_no_low
# NMT+Pred POS
# ./Main-lemm-train.sh norm_soft_char_context ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1 pos_aux_no_cont 
# NMT+Pred POS with no lowercasing
# ./Main-lemm-train.sh norm_soft_char_context ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1 pos_aux_no_cont_no_low 
# NMT+Pred POS with mask cap
# ./Main-lemm-train.sh norm_soft_char_context ud-treebanks-v2.0/UD_Basque/eu ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/eu lemm/Basque 1 pos_aux_no_cont_cap_mask 
##########################################################################################

# TODO: add beam and number of ensembles to the options

export TRAIN=$2-ud-train.conllu
export DEV=$2-ud-dev.conllu
export TEST=$3.conllu

# construct results folder name
export MODEL=$1
if [ -z $6 ]; then
export MODEL_FOLDER=$4/${MODEL}
else
export MODEL_FOLDER=$4/${MODEL}_$6
fi

export k=$5

export BEAM=3
export INPUT_FORMAT="1,2,3"

if [[ $k != 'ens' ]]; then
########### train + eval of individual models
# NMT+Context+Gold POS
if [[ $6 == "pos" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --pos_feature --conll_format --input_format=$INPUT_FORMAT
# NMT+Context+Pred POS
elif [[ $6 == "pos_aux" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --pos_feature --aux_pos_task --conll_format --input_format=$INPUT_FORMAT
# NMT+Context+Pred POS with no lowercasing
elif [[ $6 == "pos_aux_no_low" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --pos_feature --aux_pos_task --conll_format --input_format=$INPUT_FORMAT  --lowercase=False
# NMT+Pred POS
elif [[ $6 == "pos_aux_no_cont" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --pos_feature --aux_pos_task --no_context_feature --conll_format --input_format=$INPUT_FORMAT
#echo 'skip training'
elif [[ $6 == "pos_aux_no_cont_cap_mask" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --pos_feature --aux_pos_task --no_context_feature --mask_caps --lowercase=False --conll_format --input_format=$INPUT_FORMAT
elif [[ $6 == "pos_aux_no_cont_no_low" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --pos_feature --aux_pos_task --no_context_feature --conll_format --input_format=$INPUT_FORMAT  --lowercase=False
# NMT+Gold POS
elif [[ $1 == "norm_soft_pos" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --conll_format --input_format=$INPUT_FORMAT
else
# NMT or NMT+Context
# $1 == "norm_soft" or "norm_soft_context"
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --conll_format --input_format=$INPUT_FORMAT
fi

wait

if [[ $6 == *"no_low"* ]]; then
python ${MODEL}.py test ${MODEL_FOLDER}_$k --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM  --conll_format --input_format=$INPUT_FORMAT  --lowercase=False &
python ${MODEL}.py test ${MODEL_FOLDER}_$k --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM --conll_format --input_format=$INPUT_FORMAT   --lowercase=False
else
python ${MODEL}.py test ${MODEL_FOLDER}_$k --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM  --conll_format --input_format=$INPUT_FORMAT &
python ${MODEL}.py test ${MODEL_FOLDER}_$k --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM --conll_format --input_format=$INPUT_FORMAT
fi	

else

########### Evaluate ensemble 5
export MODEL_FOLDER_ENS="${MODEL_FOLDER}_1,${MODEL_FOLDER}_2,${MODEL_FOLDER}_3,${MODEL_FOLDER}_4,${MODEL_FOLDER}_5"
if [[ $6 == *"no_low"* ]]; then
python ${MODEL}.py ensemble_test ${MODEL_FOLDER_ENS} --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM ${MODEL_FOLDER}_ens5 --conll_format --input_format=$INPUT_FORMAT --lowercase=False  &
python ${MODEL}.py ensemble_test ${MODEL_FOLDER_ENS} --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM ${MODEL_FOLDER}_ens5 --conll_format --input_format=$INPUT_FORMAT --lowercase=False
else
python ${MODEL}.py ensemble_test ${MODEL_FOLDER_ENS} --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM ${MODEL_FOLDER}_ens5 --conll_format --input_format=$INPUT_FORMAT  &
python ${MODEL}.py ensemble_test ${MODEL_FOLDER_ENS} --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM ${MODEL_FOLDER}_ens5 --conll_format --input_format=$INPUT_FORMAT

fi

fi
