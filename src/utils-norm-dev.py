# Convert n-best dev files for models with HLLM to 1-best
# python utils-norm-dev.py /gennorm/normalization/results/arch-8-context-shuff/norm_soft_char_context_pos_aux_sync/ensemble/we
# python utils-norm-dev.py /gennorm/normalization/results/arch-8-context-shuff/norm_soft_char_context_pos_sync/ensemble/we
# python utils-norm-dev.py /gennorm/normalization/results/arch-8-context-shuff/norm_soft_char_context_sync/ensemble/we
# python utils-norm-dev.py /gennorm/normalization/results/arch-8-context-shuff/norm_soft_pos_sync/ensemble/we 
import sys
import codecs


file_in_nbest = sys.argv[1]+'/nbest_dev_mert.out'
file_in_src =  sys.argv[1]+'/dev.src'
file_out = sys.argv[1]+'/dev.out.predictions'

f_in_nbest= codecs.open(file_in_nbest, 'r', 'utf8')
f_in_src= codecs.open(file_in_src, 'r', 'utf8')
f_out = codecs.open(file_out, 'w', 'utf8')

dev_pred={}

count=0
for line in f_in_nbest:
    splt = line.strip().split(' ||| ')
    if count < 10:
    	print splt[0],splt[1]
    ind = int(splt[0])
    if ind==count:
    	dev_pred[count]=splt[1]
    	count+=1

for i,line in enumerate(f_in_src):
	src = line.strip()
	f_out.write(u'{}\t{}\n'.format(src,dev_pred[i]))


