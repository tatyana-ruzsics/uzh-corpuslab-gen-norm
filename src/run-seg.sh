# Training and evaluation  instructions for segmentation data TACL paper

# Example of training instruction for NMT model on split0 of English data - train models for 5 seeds end evaluate ensemble
# This is just an example - the models were already pretrained before
#./Main-seg-train.sh canonical-segmentation/english/ eng 0 1
#./Main-seg-train.sh canonical-segmentation/english/ eng 0 2
#./Main-seg-train.sh canonical-segmentation/english/ eng 0 3
#./Main-seg-train.sh canonical-segmentation/english/ eng 0 4
#./Main-seg-train.sh canonical-segmentation/english/ eng 0 5
#./Main-seg-train.sh canonical-segmentation/english/ eng 0 ens

# Training and evaluation instruction for NMT + HLLM trained on train target data+extra data
./Main-seg-sync.sh eng english 5 3 we 0
./Main-seg-sync.sh eng english 5 3 we 1
./Main-seg-sync.sh eng english 5 3 we 2
./Main-seg-sync.sh eng english 5 3 we 3
./Main-seg-sync.sh eng english 5 3 we 4
./Main-seg-sync.sh eng english 5 3 we 5
./Main-seg-sync.sh eng english 5 3 we 6
./Main-seg-sync.sh eng english 5 3 we 7
./Main-seg-sync.sh eng english 5 3 we 8
./Main-seg-sync.sh eng english 5 3 we 9
./Main-seg-sync.sh ger german 5 3 we 0
./Main-seg-sync.sh ger german 5 3 we 1
./Main-seg-sync.sh ger german 5 3 we 2
./Main-seg-sync.sh ger german 5 3 we 3
./Main-seg-sync.sh ger german 5 3 we 4
./Main-seg-sync.sh ger german 5 3 we 5
./Main-seg-sync.sh ger german 5 3 we 6
./Main-seg-sync.sh ger german 5 3 we 7
./Main-seg-sync.sh ger german 5 3 we 8
./Main-seg-sync.sh ger german 5 3 we 9

# Evaluation instruction for NMT ensembles (alternative)
./Main-seg-sync.sh eng english 5 3 nmt 0
./Main-seg-sync.sh eng english 5 3 nmt 1
./Main-seg-sync.sh eng english 5 3 nmt 2
./Main-seg-sync.sh eng english 5 3 nmt 3
./Main-seg-sync.sh eng english 5 3 nmt 4
./Main-seg-sync.sh eng english 5 3 nmt 5
./Main-seg-sync.sh eng english 5 3 nmt 6
./Main-seg-sync.sh eng english 5 3 nmt 7
./Main-seg-sync.sh eng english 5 3 nmt 8
./Main-seg-sync.sh eng english 5 3 nmt 9
./Main-seg-sync.sh ger german 5 3 nmt 0
./Main-seg-sync.sh ger german 5 3 nmt 1
./Main-seg-sync.sh ger german 5 3 nmt 2
./Main-seg-sync.sh ger german 5 3 nmt 3
./Main-seg-sync.sh ger german 5 3 nmt 4
./Main-seg-sync.sh ger german 5 3 nmt 5
./Main-seg-sync.sh ger german 5 3 nmt 6
./Main-seg-sync.sh ger german 5 3 nmt 7
./Main-seg-sync.sh ger german 5 3 nmt 8
./Main-seg-sync.sh ger german 5 3 nmt 9

# These experiments were not used for TACL paper
# Training instruction for NMT + one HLLM trained on train target data and another on extra data
#./Main-seg-sync.sh eng english 5 3 w2 0
#./Main-seg-sync.sh eng english 5 3 w2 1
#./Main-seg-sync.sh eng english 5 3 w2 2
#./Main-seg-sync.sh eng english 5 3 w2 3
#./Main-seg-sync.sh eng english 5 3 w2 4
#./Main-seg-sync.sh eng english 5 3 w2 5
#./Main-seg-sync.sh eng english 5 3 w2 6
#./Main-seg-sync.sh eng english 5 3 w2 7
#./Main-seg-sync.sh eng english 5 3 w2 8
#./Main-seg-sync.sh eng english 5 3 w2 9
#./Main-seg-sync.sh ger german 5 3 w2 0
#./Main-seg-sync.sh ger german 5 3 w2 1
#./Main-seg-sync.sh ger german 5 3 w2 2
#./Main-seg-sync.sh ger german 5 3 w2 3
#./Main-seg-sync.sh ger german 5 3 w2 4
#./Main-seg-sync.sh ger german 5 3 w2 5
#./Main-seg-sync.sh ger german 5 3 w2 6
#./Main-seg-sync.sh ger german 5 3 w2 7
#./Main-seg-sync.sh ger german 5 3 w2 8
#./Main-seg-sync.sh ger german 5 3 w2 9

# Trainingraining instruction for NMT + one HLLM trained on train target data only
#./Main-seg-sync.sh eng english 5 3 w 0
#./Main-seg-sync.sh eng english 5 3 w 1
#./Main-seg-sync.sh eng english 5 3 w 2
#./Main-seg-sync.sh eng english 5 3 w 3
#./Main-seg-sync.sh eng english 5 3 w 4
#./Main-seg-sync.sh eng english 5 3 w 5
#./Main-seg-sync.sh eng english 5 3 w 6
#./Main-seg-sync.sh eng english 5 3 w 7
#./Main-seg-sync.sh eng english 5 3 w 8
#./Main-seg-sync.sh eng english 5 3 w 9
#./Main-seg-sync.sh ger german 5 3 w 0
#./Main-seg-sync.sh ger german 5 3 w 1
#./Main-seg-sync.sh ger german 5 3 w 2
#./Main-seg-sync.sh ger german 5 3 w 3
#./Main-seg-sync.sh ger german 5 3 w 4
#./Main-seg-sync.sh ger german 5 3 w 5
#./Main-seg-sync.sh ger german 5 3 w 6
#./Main-seg-sync.sh ger german 5 3 w 7
#./Main-seg-sync.sh ger german 5 3 w 8
#./Main-seg-sync.sh ger german 5 3 w 9


