# Print training and evaluation instructions for lemmatization data TACL paper


# loop over languages
declare -a LNG=("Arabic" "Basque" "Croatian" "Portuguese"  "Estonian" "Finnish" "German" "Greek"  "Hindi" "Hungarian" "Italian" "Latvian" "Polish" "Dutch" "Romanian" "Russian" "Slovak" "Slovenian" "Turkish" "Urdu")
declare -a PF=("ar" "eu" "hr" "pt" "et"  "fi" "de" "el"  "hi" "hu" "it" "lv" "pl" "nl" "ro" "ru" "sk" "sl" "tr" "ur")
# for testing
#declare -a LNG=("Arabic")
#declare -a PF=("ar")

# get length of an array
arraylength=${#LNG[@]}

for (( i=1; i<${arraylength}+1; i++ ));
do
export l=${LNG[$i-1]}
export pf=${PF[$i-1]}

if [[ $l == 'Dutch' ]]; then
export UDTR='ud-treebanks-v2.1/UD_'
export UDTE='ud-treebanks-v2.1/UD_'
export DATATE=${UDTR}${l}/${pf}
else
export UDTR='ud-treebanks-v2.0/UD_'
export UDTE='ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/'
export DATATE=${UDTE}${pf}
fi

export DATATR=${UDTR}${l}/${pf}
export RESULTSPF='lemm'
export RESULTS=${RESULTSPF}/${l}

if [[ $1 == 'train' ]]; then
#########################################################
# Train instructions
#########################################################
# loop over seeds
for (( k=1; k<=5; k++ ))
do
# NMT+Gold POS
echo ./Main-lemm-train.sh norm_soft_pos $DATATR $DATATE $RESULTS $k
# NMT+Context
echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k
# NMT+Context+Gold POS
echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos
# NMT+Context+Pred POS
echo  ./Main-lemm-train.sh norm_soft_char_context $DATATR $DATATE $RESULTS $k pos_aux
done

else
#########################################################
# Evaluation instructions
#########################################################
# Baseline and Baseline+POS
echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS baseline

# ensemble of 5 source context NMT models
# NMT+Gold POS
echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_pos ens 5 3
# NMT+Context
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3
# NMT+Context+Gold POS
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos
# NMT+Context+Pred POS
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ens 5 3 pos_aux

# individual source context NMT models
for (( k=1; k<=5; k++ ))
do
# NMT+Gold POS
echo ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_pos ind $k 3
# NMT+Context
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3
# NMT+Context+Gold POS
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos
# NMT+Context+Pred POS
echo  ./Main-lemm-eval.sh  $DATATR $DATATE $RESULTS norm_soft_char_context ind $k 3 pos_aux
done

fi
done
