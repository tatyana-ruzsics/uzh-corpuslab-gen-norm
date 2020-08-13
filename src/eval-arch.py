
import glob
import sys
import re
import numpy as np

PREFIX = sys.argv[1] # arch-8-context-shuff

results = {}
pos_baseline = {}
#models = ['norm_soft', 'norm_soft_char_context','norm_soft_char_context_pos','norm_soft_char_context_pos_aux','norm_soft_pos']
models = ['norm_soft_char_context','norm_soft_char_context_pos','norm_soft_char_context_pos_aux','norm_soft_pos']
model_names = ['nmt','context','context_pos','context_pos_aux','pos']
datasets = ['dev']
#settings = ['nmt', 'we_t', 'we']
settings = ['nmt', 'we']
#setting_names  = ['Arch', 'Arch+SMS-data', 'Arch+OpenSub-data' ]
setting_names  = ['Arch', 'Arch+OpenSub-data' ]
data_cats = ['total', 'unique', 'new', 'ambiguous', 'ambiguous_pos_unamb', 'ambiguous_pos_amb']
data_cats_names = ['Total', 'Unamb.', 'New', 'Amb.', 'POS-unamb.', 'POS-amb.']
# we - with big lm, we_t, nmt_t - with small lm
for data in datasets:
    results[data] = {}
    pos_baseline[data] = {}
    for cat in data_cats:
        pos_baseline[data][cat] = {}
#    print('../results/{}/baseline/baseline-pos.{}.eval'.format(PREFIX,data))
    result_path = glob.glob('/gennorm/normalization/results/{}/baseline/baseline-pos.{}.eval'.format(PREFIX,data))
    with open(result_path[0]) as f:
        for line in f:
            m_total = re.search('(\s+)Number\sof\scorrect\spredictions\stotal:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_new = re.search('(\s+)-\snew:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_un = re.search('(\s+)-\sunique:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_amb = re.search('(\s+)-\sambigous:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_amb_pos_unamb = re.search('(\s+)-\sambigous\swith\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_amb_pos_amb = re.search('(\s+)-\sambigous\swith\sno\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            
            m_new_s = re.search('(.*)new\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_un_s = re.search('(.*)unique\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_amb_s = re.search('(.*)ambigous\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_amb_pos_unamb_s = re.search('(\s+)-\scan\sbe\sPOS\sdisambiguated:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_amb_pos_amb_s = re.search('(\s+)-\scannot\sbe\sPOS\sdisambiguated:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            # preformance
            if m_total:
                pos_baseline[data]['total']['perf'] = float(m_total.group(5))
            if m_un:
                pos_baseline[data]['unique']['perf'] = float(m_un.group(5))
            if m_new:
                pos_baseline[data]['new']['perf'] = float(m_new.group(5))
            if m_amb:
                pos_baseline[data]['ambiguous']['perf'] = float(m_amb.group(5))
            if m_amb_pos_unamb:
                pos_baseline[data]['ambiguous_pos_unamb']['perf'] = float(m_amb_pos_unamb.group(5))
            if m_amb_pos_amb:
                pos_baseline[data]['ambiguous_pos_amb']['perf'] = float(m_amb_pos_amb.group(5))
            # categories statistics
            if m_un_s:
                 pos_baseline[data]['unique']['stat'] = float(m_un_s.group(5))
            if m_new_s:
                pos_baseline[data]['new']['stat'] = float(m_new_s.group(5))
            if m_amb_s:
                pos_baseline[data]['ambiguous']['stat'] = float(m_amb_s.group(5))
            if m_amb_pos_unamb_s:
                pos_baseline[data]['ambiguous_pos_unamb']['stat'] = float(m_amb_pos_unamb_s.group(5))
            if m_amb_pos_amb_s:
                pos_baseline[data]['ambiguous_pos_amb']['stat'] = float(m_amb_pos_amb_s.group(5))
    for model in models:
#            print('Model {:>20s}:'.format(model))
        results[data][model] = {}
        for setting in settings:
#            print('Folder {}:'.format(setting))
            results[data][model][setting] = {}
#            print('{:>30s}'.format('{} set:'.format(data)))
#            print('{:>30s}'.format('ensemble:'))
            print '/gennorm/normalization/results/{}/{}_sync/ensemble/{}/{}.eval.det'.format(PREFIX,model,setting,data)
            result_path = glob.glob('/gennorm/normalization/results/{}/{}_sync/ensemble/{}/{}.eval.det'.format(PREFIX,model,setting,data))
            print(result_path[0])
            with open(result_path[0]) as f:
                for line in f:
                    #print(line)
                    m_total = re.search('(\s+)Number\sof\scorrect\spredictions\stotal:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_amb = re.search('(\s+)-\sambigous:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_new = re.search('(\s+)-\snew:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_un = re.search('(\s+)-\sunique:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    if m_total:
                        results[data][model][setting]['total'] = float(m_total.group(5))
                    if m_amb:
                        results[data][model][setting]['ambiguous'] = float(m_amb.group(5))
                    if m_new:
                        results[data][model][setting]['new'] = float(m_new.group(5))
                    if m_un:
                        results[data][model][setting]['unique'] = float(m_un.group(5))
            result_path = glob.glob('/gennorm/normalization/results/{}/{}_sync/ensemble/{}/{}.eval.det.pos'.format(PREFIX,model,setting,data))
            #print(result_path[0])
            with open(result_path[0]) as f:
                for line in f:
                    m_amb_pos_unamb = re.search('(\s+)-\sambigous\swith\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_amb_pos_amb = re.search('(\s+)-\sambigous\swith\sno\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    if m_amb_pos_unamb:
                        results[data][model][setting]['ambiguous_pos_unamb'] = float(m_amb_pos_unamb.group(5))
                    if m_amb_pos_amb:
                        results[data][model][setting]['ambiguous_pos_amb'] = float(m_amb_pos_amb.group(5))

def max_markup(l):
    #    print(l)
    max_index = l.index(max(l))
    markup = ['']*len(l)
    markup[max_index] = '*'
    return '\t'.join( '{:.2f}'.format(i1)+i2 for (i1,i2) in zip(l,markup) )
        
print('{}'.format('\t'.join(['Category','No of','POS-baseline'] + model_names)))
for data in datasets:
    print('{}:'.format(data))
    for i,setting in enumerate(settings):
        print('{}:'.format(setting_names[i]))
        for j,cat in enumerate(data_cats):
            results_ = [results[data][m][setting][cat] for m in models]
            if cat!='total':
                print('{}\t{}\t'.format(data_cats_names[j],pos_baseline[data][cat]['stat'])+max_markup([pos_baseline[data][cat]['perf']] + results_))
            else:
                print('{}\t{}\t'.format(data_cats_names[j], '')+max_markup([pos_baseline[data][cat]['perf']] + results_))
        
#        results_total = [results[data][m][setting]['total'] for m in models]
#        print('{}\t'.format('total')+max_markup([pos_baseline[data]['total']] + results_total))
#        results_amb = [results[data][m][setting]['ambiguous'] for m in models]
#        print('{}\t'.format('amb')+max_markup([pos_baseline[data]['ambiguous']] + results_amb))
#        results_amb_pos_unamb = [results[data][m][setting]['ambiguous_pos_unamb'] for m in models]
#        print('{}\t'.format('POS-unamb')+max_markup([pos_baseline[data]['ambiguous_pos_unamb']] + results_amb_pos_unamb))
#        results_amb_pos_amb = [results[data][m][setting]['ambiguous_pos_amb'] for m in models]
#        print('{}\t'.format('POS-amb')+max_markup([pos_baseline[data]['ambiguous_pos_amb']] + results_amb_pos_amb))
#        results_uniq = [results[data][m][setting]['unique'] for m in models]
#        print('{}\t'.format('unique')+max_markup([pos_baseline[data]['unique']] + results_uniq))
#        results_new = [results[data][m][setting]['new'] for m in models]
#        print('{}\t'.format('new')+max_markup([pos_baseline[data]['new']] + results_new))
#            print('{:>40s}: {}'.format('total', acc_total ))
#            print('{:>40s}: {}'.format('ambiguous', acc_amb ))
#            print('{:>40s}: {}'.format('POS-unamb', acc_amb_pos_unamb ))
#            print('{:>40s}: {}'.format('POS-amb', acc_amb_pos_amb ))
#            print('{:>40s}: {}'.format('unique', acc_un ))
#            print('{:>40s}: {}'.format('new', acc_new ))





