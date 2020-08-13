# Print training and evaluation instructions for lemmatization data TACL paper


# loop over languages
declare -a LNG=("Arabic" "Basque" "Croatian" "Portuguese"  "Estonian" "Finnish" "German" "Greek"  "Hindi" "Hungarian" "Italian" "Latvian" "Polish" "Dutch" "Romanian" "Russian" "Slovak" "Slovenian" "Turkish" "Urdu")
declare -a PF=("ar" "eu" "hr" "pt" "et"  "fi" "de" "el"  "hi" "hu" "it" "lv" "pl" "nl" "ro" "ru" "sk" "sl" "tr" "ur")
# for testing
#declare -a LNG=("Arabic")
#declare -a PF=("ar")


# get length of an array
arraylength=${#LNG[@]}

export DATAPF='/gennorm/lemmatization/data/'
export RESULTSPF='/gennorm/lemmatization/'

for (( i=1; i<${arraylength}+1; i++ ));
do
export l=${LNG[$i-1]}
export pf=${PF[$i-1]}

if [[ $l == 'Dutch' ]]; then
export UDTR='ud-treebanks-v2.1/UD_'
export UDTE='ud-treebanks-v2.1/UD_'
export DATATE=${DATAPF}${UDTR}${l}/${pf}
else
export UDTR='ud-treebanks-v2.0/UD_'
export UDTE='ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/'
export DATATE=${DATAPF}${UDTE}${pf}
fi

export DATATR=${DATAPF}${UDTR}${l}/${pf}

export RESULTS=${RESULTSPF}/${l}

if [[ $1 == 'train' ]]; then
#########################################################
# Train instructions
#########################################################
# loop over seeds
for (( k=2; k<=2; k++ ))
do
# NMT+Gold POS
# echo ./Main-lemm-train.sh norm_soft_pos $DATATR $DATATE $RESULTS $k
 echo ./Main-lemm-train.sh norm_soft_pos $DATATR $DATATE $RESULTS $k no_low
# # NMT+Context
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k
#echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k no_low
# # NMT+Context+Gold POS
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos_no_low
# # NMT+Context+Pred POS
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos_aux
# # NMT+Context+Pred POS and no lowercasing
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos_aux_no_low
# NMT+Pred POS
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos_aux_no_cont
# NMT+Pred POS and no lowercasing
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos_aux_no_cont_no_low
# NMT+Pred POS+mask and copy
# echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos_aux_no_cont_cap_mask

done

else
#########################################################
# Evaluation instructions
#########################################################
# Baseline and Baseline+POS
# echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS baseline

# ensemble of 5 source context NMT models
# NMT+Gold POS
# echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_pos ens 5 3
echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_pos ens 5 3 no_low
# # NMT+Context
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 no_low
# # NMT+Context+Gold POS
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos_no_low
# # NMT+Context+Pred POS
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos_aux
# # NMT+Context+Pred POS and no lowercasing
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos_aux_no_low
# NMT+Pred POS
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos_aux_no_cont
# NMT+Pred POS and no lowercasing
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos_aux_no_cont_no_low
# NMT+Pred POS+mask and copy
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos_aux_no_cont_cap_mask

# individual source context NMT models
#for (( k=2; k<=2; k++ ))
#do
# NMT+Gold POS
# echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_pos ind $k 3
# echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_pos ind $k 3 no_low
# # NMT+Context
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 no_low
# # NMT+Context+Gold POS
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos_no_low
# # NMT+Context+Pred POS
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos_aux
# # NMT+Context+Pred POSand no lowercasing
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos_aux_no_low
# NMT+Pred POS
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos_aux_no_cont
# NMT+Pred POS and no lowercasing
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos_aux_no_cont_no_low
# NMT+Pred POS+mask and copy
# echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos_aux_no_cont_cap_mask
#done

fi
done
