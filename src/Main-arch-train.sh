#!/bin/bash
# Usage Main-arch-train.sh model_file_name data_folder results_folder_name seed/ens use_aux_loss_and/or_pos_feat
# NMT
# ./Main-arch-train.sh norm_soft Archimob-pos arch 1
# NMT+Gold POS
# ./Main-arch-train.sh norm_soft_pos Archimob-pos arch 1
# NMT+Context
# ./Main-arch-train.sh norm_soft_char_context Archimob-pos arch 1
# NMT+Context+Gold POS
# ./Main-arch-train.sh norm_soft_char_context Archimob-pos arch 1 pos
# NMT+Context+Pred POS
# ./Main-arch-train.sh norm_soft_char_context Archimob-pos arch 1 pos_aux
##########################################################################################

# TODO: add beam and number of ensembles to the options

export TRAIN=$2/train.txt
export DEV=$2/dev.txt
export TEST=$2/test.txt

# construct results folder name
export MODEL=$1
if [ -z $5 ]; then
export MODEL_FOLDER=$3/${MODEL}
else
export MODEL_FOLDER=$3/${MODEL}_$5
fi

export BEAM=3
export k=$4

if [[ $k != 'ens' ]]; then
########### train + eval of individual models NMT models
# NMT+Context+Gold POS
if [[ $5 == "pos" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --epochs=40  --pos_feature
# NMT+Context+Pred POS
elif [[ $5 == "pos_aux" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --epochs=40  --pos_feature --aux_pos_task
# NMT+Gold POS
elif [[ $1 == "norm_soft_pos" ]]; then
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --epochs=40
else
# NMT or NMT+Context
# $1 == "norm_soft" or "norm_soft_context"
python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${MODEL_FOLDER}_$k  --epochs=40
fi

wait

python ${MODEL}.py test ${MODEL_FOLDER}_$k --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM   &
python ${MODEL}.py test ${MODEL_FOLDER}_$k --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM


else
########### Evaluate ensemble of 5 NMT models
export MODEL_FOLDER_ENS="${MODEL_FOLDER}_1,${MODEL_FOLDER}_2,${MODEL_FOLDER}_3,${MODEL_FOLDER}_4,${MODEL_FOLDER}_5"
python ${MODEL}.py ensemble_test ${MODEL_FOLDER_ENS} --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM ${MODEL_FOLDER}_ens5   &
python ${MODEL}.py ensemble_test ${MODEL_FOLDER_ENS} --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM ${MODEL_FOLDER}_ens5

fi
