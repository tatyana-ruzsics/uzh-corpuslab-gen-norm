# Training and evaluation instructions for normalization data TACL paper


export DATA='Archimob-8-context'
export RESULTS='arch-8-context-shuff'

#########################################################
# NMT
#########################################################
# train
#./Main-arch-train.sh norm_soft $DATA $RESULTS 1
#./Main-arch-train.sh norm_soft $DATA $RESULTS 2
#./Main-arch-train.sh norm_soft $DATA $RESULTS 3
#./Main-arch-train.sh norm_soft $DATA $RESULTS 4
#./Main-arch-train.sh norm_soft $DATA $RESULTS 5

# detailed eval
#./Main-arch-sync.sh $DATA $RESULTS 5 3 nmt norm_soft
# sync decoding
#./Main-arch-sync.sh $DATA $RESULTS 5 3 we norm_soft


#########################################################
# NMT+Gold POS
#########################################################

# train
#./Main-arch-train.sh norm_soft_pos $DATA $RESULTS 1
#./Main-arch-train.sh norm_soft_pos $DATA $RESULTS 2
#./Main-arch-train.sh norm_soft_pos $DATA $RESULTS 3
#./Main-arch-train.sh norm_soft_pos $DATA $RESULTS 4
#./Main-arch-train.sh norm_soft_pos $DATA $RESULTS 5

# detailed eval
#./Main-arch-sync.sh $DATA $RESULTS 5 3 nmt norm_soft_pos
# sync decoding
#./Main-arch-sync.sh $DATA $RESULTS 5 3 we norm_soft_pos

#########################################################
# NMT+Context, NMT+Context+Gold POS, NMT+Context+Pred POS
#########################################################

# train
# NMT+Context
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 1
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 2
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 3
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 4
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 5
# NMT+Context+Gold POS
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 1 pos
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 2 pos
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 3 pos
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 4 pos
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 5 pos
# NMT+Context+Pred POS
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 1 pos_aux
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 2 pos_aux
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 3 pos_aux
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 4 pos_aux
#./Main-arch-train.sh norm_soft_char_context $DATA $RESULTS 5 pos_aux

# NMT+Context
# detailed eval
#./Main-arch-sync.sh $DATA $RESULTS 5 3 nmt norm_soft_char_context
# sync decoding
#./Main-arch-sync.sh $DATA $RESULTS 5 3 we norm_soft_char_context


# NMT+Context+Gold POS
# detailed eval
#./Main-arch-sync.sh $DATA $RESULTS 5 3 nmt norm_soft_char_context pos
# sync decoding
#./Main-arch-sync.sh $DATA $RESULTS 5 3 we norm_soft_char_context pos


# NMT+Context+Pred POS
# detailed eval
#./Main-arch-sync.sh $DATA $RESULTS 5 3 nmt norm_soft_char_context pos_aux
# sync decoding
#./Main-arch-sync.sh $DATA $RESULTS 5 3 we norm_soft_char_context pos_aux


#########################################################
# Baseline and Baseline+POS
#########################################################

#export DIR=/home/tanja/uzh-corpuslab-gen-norm
#export RESULTS_BASELINE=$DIR/results/$RESULTS/baseline
#mkdir -p ${RESULTS_BASELINE}
#export TRAINDATA=$DIR/data/$DATA/train.txt
#export DEVDATA=$DIR/data/$DATA/dev.txt
#export TESTDATA=$DIR/data/$DATA/test.txt
#
## eval Baseline
#python accuracy-det.py eval_baseline $TRAINDATA $DEVDATA --error_file=${RESULTS_BASELINE}/Errors_baseline_dev.txt > ${RESULTS_BASELINE}/baseline.dev.eval
#python accuracy-det.py eval_baseline $TRAINDATA $TESTDATA --error_file=${RESULTS_BASELINE}/Errors_baseline_test.txt > ${RESULTS_BASELINE}/baseline.test.eval
#
## eval Baseline+POS
#python accuracy-det.py eval_ambiguity_baseline $TRAINDATA $DEVDATA --error_file=${RESULTS_BASELINE}/Errors_baseline_pos_dev.txt > ${RESULTS_BASELINE}/baseline-pos.dev.eval &
#python accuracy-det.py eval_ambiguity_baseline $TRAINDATA $TESTDATA --error_file=${RESULTS_BASELINE}/Errors_baseline_pos_test.txt > ${RESULTS_BASELINE}/baseline-pos.test.eval

