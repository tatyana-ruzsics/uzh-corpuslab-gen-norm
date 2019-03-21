#!/bin/bash
# Usage Main-train.sh DataFolder ResultPrefix DataSplit NMTSeed/Ensemble
# ./Main-seg-train.sh canonical-segmentation/english/ eng 0 1
# ./Main-seg-train.sh canonical-segmentation/english/ eng 0 ens
# ./Main-seg-train.sh canonical-segmentation/indonesian/ ind 0 1
# ./Main-seg-train.sh canonical-segmentation/german/ ger 0 1
##########################################################################################

# TODO: add beam and number of ensembles to the options

export DATA=$1

export n=$3
export k=$4


export TRAIN=$DATA/train$n
export DEV=$DATA/dev$n
export TEST=$DATA/test$n

export PR=$2/$2_$n
echo "$PR"

export BEAM=3
export INPUT_FORMAT="0,2,1"

if [[ $k != 'ens' ]]; then
########### train + eval of individual NMT models

python norm_soft.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_$k --input_format=$INPUT_FORMAT

wait

python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM --input_format=$INPUT_FORMAT &
python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM --input_format=$INPUT_FORMAT

else
########### Evaluate ensemble of 5 NMT models

python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=$BEAM --pred_path=best.dev.$BEAM ${PR}_nmt_ens5 --input_format=$INPUT_FORMAT &
python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=$BEAM --pred_path=best.test.$BEAM ${PR}_nmt_ens5 --input_format=$INPUT_FORMAT


fi
