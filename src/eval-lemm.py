import glob
import sys
import re
import numpy as np

#PREFIX = sys.argv[1] # lemm/Polish

baseline = {}
pos_baseline = {}
results = {}
stats = {}
models = ['norm_soft_char_context','norm_soft_char_context_pos','norm_soft_char_context_pos_aux','norm_soft_pos']
model_names = ['context','context_pos','context_pos_aux','pos']
langs = ["Arabic", "Basque", "Croatian", "Dutch", "Estonian", "Finnish", "German", "Greek",  "Hindi", "Hungarian", "Italian", "Latvian", "Polish", "Portuguese", "Romanian", "Russian", "Slovak", "Slovenian", "Turkish", "Urdu"]
datasets = ['dev', 'test']
data_cats = ['total', 'unique', 'new', 'ambiguous', 'ambiguous_pos_unamb', 'ambiguous_pos_amb']
data_cats_names = ['Total', 'Unamb.', 'New', 'Amb.', 'POS-unamb.', 'POS-amb.']
for data in datasets:
    baseline[data] = {}
    stats[data] = {}
    for lang in langs:
        baseline[data][lang] = {}
        stats[data][lang] = {}
        result_path = glob.glob('/gennorm/lemmatization/{}/baseline/baseline.{}.eval'.format(lang,data))
        #print(result_path[0])
        with open(result_path[0]) as f:
            for line in f:
                #print(line)
                m_total = re.search('(\s+)Number\sof\scorrect\spredictions\stotal:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_amb = re.search('(\s+)-\sambigous:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_new = re.search('(\s+)-\snew:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_un = re.search('(\s+)-\sunique:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                
                m_new_s = re.search('(.*)new\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_un_s = re.search('(.*)unique\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_amb_s = re.search('(.*)ambigous\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                # preformance
                if m_total:
                    baseline[data][lang]['total'] = float(m_total.group(5))
                if m_amb:
                    baseline[data][lang]['ambiguous'] = float(m_amb.group(5))
                if m_un:
                    baseline[data][lang]['unique'] = float(m_un.group(5))
                if m_new:
                    baseline[data][lang]['new'] = float(m_new.group(5))
                # categories statistics
                if m_un_s:
                    stats[data][lang]['unique'] = float(m_un_s.group(5))
                if m_new_s:
                    stats[data][lang]['new'] = float(m_new_s.group(5))
                if m_amb_s:
                    stats[data][lang]['ambiguous'] = float(m_amb_s.group(5))
    pos_baseline[data] = {}
    for lang in langs:
        pos_baseline[data][lang] = {}
        result_path = glob.glob('/gennorm/lemmatization/{}/baseline/baseline-pos.{}.eval'.format(lang,data))
        with open(result_path[0]) as f:
            for line in f:
                m_total = re.search('(\s+)Number\sof\scorrect\spredictions\stotal:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_amb = re.search('(\s+)-\sambigous:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_amb_pos_unamb = re.search('(\s+)-\sambigous\swith\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_amb_pos_amb = re.search('(\s+)-\sambigous\swith\sno\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                
                m_amb_pos_unamb_s = re.search('(\s+)-\scan\sbe\sPOS\sdisambiguated:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_amb_pos_amb_s = re.search('(\s+)-\scannot\sbe\sPOS\sdisambiguated:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                # performance
                if m_total:
                    pos_baseline[data][lang]['total'] = float(m_total.group(5))
                if m_amb:
                    pos_baseline[data][lang]['ambiguous'] = float(m_amb.group(5))
                if m_amb_pos_unamb:
                    pos_baseline[data][lang]['ambiguous_pos_unamb'] = float(m_amb_pos_unamb.group(5))
                if m_amb_pos_amb:
                    pos_baseline[data][lang]['ambiguous_pos_amb'] = float(m_amb_pos_amb.group(5))
                
                pos_baseline[data][lang]['unique'] = baseline[data][lang]['unique']
                pos_baseline[data][lang]['new'] = baseline[data][lang]['new']
                # categories statistics
                if m_amb_pos_unamb_s:
                    stats[data][lang]['ambiguous_pos_unamb'] = float(m_amb_pos_unamb_s.group(5))
                if m_amb_pos_amb_s:
                    stats[data][lang]['ambiguous_pos_amb'] = float(m_amb_pos_amb_s.group(5))
    
    results[data] = {}
    for model in models:
        results[data][model] = {}
        for lang in langs:
            results[data][model][lang] = {}
            results[data][model][lang]['ens'] = {}
            results[data][model][lang]['seed_ave'] = {}
            for category in ['total', 'ambiguous', 'unique', 'new']:
                results[data][model][lang]['seed_ave'][category] = []
            for seed in range(5,6):
                #print('/gennorm/lemmatization/{}/{}_{}/{}.eval.det'.format(lang,model,seed,data))
                result_path = glob.glob('/gennorm/lemmatization/{}/{}_{}/{}.eval.det'.format(lang,model,seed,data))
                #print(result_path[0])
                with open(result_path[0]) as f:
                    for line in f:
                        #print(line)
                        m_total = re.search('(\s+)Number\sof\scorrect\spredictions\stotal:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                        m_amb = re.search('(\s+)-\sambigous:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                        m_new = re.search('(\s+)-\snew:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                        m_un = re.search('(\s+)-\sunique:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                        if m_total:
                            results[data][model][lang]['seed_ave']['total'].append(float(m_total.group(5)))
                        if m_amb:
                            results[data][model][lang]['seed_ave']['ambiguous'].append(float(m_amb.group(5)))
                        if m_new:
                            results[data][model][lang]['seed_ave']['new'].append(float(m_new.group(5)))
                        if m_un:
                            results[data][model][lang]['seed_ave']['unique'].append(float(m_un.group(5)))
            for category in ['total','ambiguous','unique','new']:
                values = np.array(results[data][model][lang]['seed_ave'][category])
                mean = np.mean(values)
                results[data][model][lang]['seed_ave'][category] = mean

            #print('/gennorm/lemmatization/{}/{}_ens5/{}.out.eval'.format(lang,model,data))
            result_path = glob.glob('/gennorm/lemmatization/{}/{}_ens5/{}.eval.det'.format(lang,model,data))
            #print(result_path[0])
            with open(result_path[0]) as f:
                for line in f:
                    #print(line)
                    m_total = re.search('(\s+)Number\sof\scorrect\spredictions\stotal:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_amb = re.search('(\s+)-\sambigous:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_new = re.search('(\s+)-\snew:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_un = re.search('(\s+)-\sunique:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    if m_total:
                        results[data][model][lang]['ens']['total'] = float(m_total.group(5))
                    if m_amb:
                        results[data][model][lang]['ens']['ambiguous'] = float(m_amb.group(5))
                    if m_un:
                        results[data][model][lang]['ens']['unique'] = float(m_un.group(5))
                    if m_new:
                        results[data][model][lang]['ens']['new'] = float(m_new.group(5))
            # ambiguity statistics
            result_path = glob.glob('/gennorm/lemmatization/{}/{}_ens5/{}.eval.det.pos'.format(lang,model,data))
            with open(result_path[0]) as f:
                for line in f:
                    m_amb_pos_unamb = re.search('(\s+)-\sambigous\swith\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    m_amb_pos_amb = re.search('(\s+)-\sambigous\swith\sno\sPOS\sdisambig:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                    if m_amb_pos_unamb:
                        results[data][model][lang]['ens']['ambiguous_pos_unamb'] = float(m_amb_pos_unamb.group(5))
                    if m_amb_pos_amb:
                        results[data][model][lang]['ens']['ambiguous_pos_amb'] = float(m_amb_pos_amb.group(5))


def max_markup(l):
#    print(l)
    max_index = l.index(max(l))
    markup = ['']*len(l)
#    markup[max_index] = '*'
    markup[max_index] = ''
    return '\t'.join( '{:.2f}'.format(i1)+i2 for (i1,i2) in zip(l,markup) )

#results_matrix_total = np.zeros((2, len(langs), len(models)))
#results_matrix_amb = np.zeros((2, len(langs), len(models)))
#results_matrix_un = np.zeros((2, len(langs), len(models)))
#results_matrix_new = np.zeros((2, len(langs), len(models)))
#results_matrix_amb_pos_unamb = np.zeros((2, len(langs), len(models)))
#results_matrix_amb_pos_amb = np.zeros((2, len(langs), len(models)))
#print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('','baseline', 'pos-baseline', *model_names))
print('{}'.format('\t'.join(['Category','No of','baseline','POS-baseline'] + model_names)))
for data in datasets:
    if data == 'dev':
        print('{}:'.format(data.upper()))
        print('Ensemble results per language:')
        for i,lang in enumerate(langs):
            for j,cat in enumerate(data_cats):
                results_ = [results[data][m][lang]['ens'][cat] for m in models]
                if cat=='total':
                    print('{}\t{}\t'.format(lang.upper(),'')+max_markup([baseline[data][lang][cat],pos_baseline[data][lang][cat]] + results_))
                else:
                    if cat not in ['ambiguous_pos_unamb', 'ambiguous_pos_amb']:
                        print('{}\t{}\t'.format(data_cats_names[j],stats[data][lang][cat])+max_markup([baseline[data][lang][cat],pos_baseline[data][lang][cat]] + results_))
                    else:
                        print('{}\t{}\t{}\t'.format(data_cats_names[j],stats[data][lang][cat],'-')+max_markup([pos_baseline[data][lang][cat]] + results_))
print
#        results_matrix_total[0,i,:] = np.array([results[data][m][lang]['ens']['total'] for m in models])
#        results_matrix_total[1,i,:] = np.array([results[data][m][lang]['seed_ave']['total'] for m in models])
#        print('{} '.format(lang)+max_markup([baseline[data][lang]['total']] + [pos_baseline[data][lang]['total']] + results_matrix_total[0,i,:].tolist()))
#        print('{:>10} {:>15.1f} {:>15.1f} {:>15.1f} {:>15.1f} {:>15.1f}'.format(lang,baseline[data][lang]['total'], *results_matrix_total[0,i,:]))
#
#        results_matrix_amb[0,i,:] = np.array([results[data][m][lang]['ens']['ambiguous'] for m in models])
#        results_matrix_amb[1,i,:] = np.array([results[data][m][lang]['seed_ave']['ambiguous'] for m in models])
#
#        results_matrix_un[0,i,:] = np.array([results[data][m][lang]['ens']['unique'] for m in models])
#        results_matrix_un[1,i,:] = np.array([results[data][m][lang]['seed_ave']['unique'] for m in models])
#
#        results_matrix_new[0,i,:] = np.array([results[data][m][lang]['ens']['new'] for m in models])
#        results_matrix_new[1,i,:] = np.array([results[data][m][lang]['seed_ave']['new'] for m in models])
#
#        results_matrix_amb_pos_unamb[0,i,:] = np.array([results[data][m][lang]['ens']['ambiguous_pos_unamb'] for m in models])
#        results_matrix_amb_pos_amb[0,i,:] = np.array([results[data][m][lang]['ens']['ambiguous_pos_amb'] for m in models])
#
#        print('{}\t'.format('amb')+max_markup([baseline[data][lang]['ambiguous']] + [pos_baseline[data][lang]['ambiguous']] + results_matrix_amb[0,i,:].tolist()))
#        print('{}\t{}\t'.format('POS-unamb','')+max_markup([pos_baseline[data][lang]['ambiguous_pos_unamb']] + results_matrix_amb_pos_unamb[0,i,:].tolist()))
#        print('{}\t{}\t'.format('POS-amb','')+max_markup([pos_baseline[data][lang]['ambiguous_pos_amb']] + results_matrix_amb_pos_amb[0,i,:].tolist()))

# Individual results - averages over langs
print('{}'.format('\t'.join(['Category','Baseline','POS-baseline'] + model_names)))
for data in datasets:
    print('{}:'.format(data.upper()))
    print('Single model averages over 20 languages:')
    for j,cat in enumerate(data_cats):
        if cat not in ['ambiguous_pos_unamb', 'ambiguous_pos_amb']:
            results_matrix = np.zeros((len(langs), len(models)))
            for k,lang in enumerate(langs):
                results_matrix[k,:] = np.array([results[data][m][lang]['seed_ave'][cat] for m in models])
            results_per_lang_ave_ = np.mean(results_matrix, 0)
            print('{}\t{}\t{}\t'.format(data_cats_names[j],'-','-')+max_markup(results_per_lang_ave_.tolist()))
#    results_per_lang_total_ave = np.mean(results_matrix_total[1,:,:],0)
#    results_per_lang_amb_ave = np.mean(results_matrix_amb[1,:,:],0)
#    results_per_lang_un_ave = np.mean(results_matrix_un[1,:,:],0)
#    results_per_lang_new_ave = np.mean(results_matrix_new[1,:,:],0)
#
#    print('{}'.format('-'*90))
#    print('Averaged results over individual models:')
    
                      
#    print('{}\t{}\t{}\t'.format('Amb.','-','-')+max_markup(results_per_lang_amb_ave.tolist()))
#    print('{}\t{}\t{}\t'.format('New','-','-')+max_markup(results_per_lang_new_ave.tolist()))
#    print('{}\t{}\t{}\t'.format('Unamb.','-','-')+max_markup(results_per_lang_un_ave.tolist()))
#    print('{}\t{}\t{}\t'.format('Total','-','-')+max_markup(results_per_lang_total_ave.tolist()))

# Ensemble results + baseline - averages over langs
                  

#    results_per_lang_total_ave = np.mean(results_matrix_total[0,:,:],0)
#    results_per_lang_amb_ave = np.mean(results_matrix_amb[0,:,:],0)
#    results_per_lang_un_ave = np.mean(results_matrix_un[0,:,:],0)
#    results_per_lang_new_ave = np.mean(results_matrix_new[0,:,:],0)
#    results_per_lang_amb_pos_unamb_ave = np.mean(results_matrix_amb_pos_unamb[0,:,:],0)
#    results_per_lang_amb_pos_amb_ave = np.mean(results_matrix_amb_pos_amb[0,:,:],0)
#
#    baseline_total_ave = np.mean(np.array([baseline[data][l]['total'] for l in langs]))
#    baseline_amb_ave = np.mean(np.array([baseline[data][l]['ambiguous'] for l in langs]))
#    baseline_un_ave = np.mean(np.array([baseline[data][l]['unique'] for l in langs]))
#    baseline_new_ave = np.mean(np.array([baseline[data][l]['new'] for l in langs]))
#
#    pos_baseline_total_ave = np.mean(np.array([pos_baseline[data][l]['total'] for l in langs]))
#    pos_baseline_amb_ave = np.mean(np.array([pos_baseline[data][l]['ambiguous'] for l in langs]))
#    pos_baseline_amb_pos_unamb_ave = np.mean(np.array([pos_baseline[data][l]['ambiguous_pos_unamb'] for l in langs]))
#    pos_baseline_amb_pos_amb_ave = np.mean(np.array([pos_baseline[data][l]['ambiguous_pos_amb'] for l in langs]))
#    pos_baseline_un_ave = baseline_un_ave
#    pos_baseline_new_ave = baseline_new_ave

#    print('{}'.format('-'*90))
    print('Ensemble model averages over 20 languages:')
    for j,cat in enumerate(data_cats):
        results_matrix = np.zeros((len(langs), len(models)))
        for k,lang in enumerate(langs):
            results_matrix[k,:] = np.array([results[data][m][lang]['ens'][cat] for m in models])
        results_per_lang_ave_ = np.mean(results_matrix, 0)
        pos_baseline_ave = np.mean(np.array([pos_baseline[data][l][cat] for l in langs]))
        if cat not in ['ambiguous_pos_unamb', 'ambiguous_pos_amb']:
            baseline_ave = np.mean(np.array([baseline[data][l][cat] for l in langs]))
            print('{}\t'.format(data_cats_names[j])+max_markup([baseline_ave, pos_baseline_ave] + results_per_lang_ave_.tolist()))
        else:
            baseline_ave = '-'
            print('{}\t{}\t'.format(data_cats_names[j], '-')+max_markup([pos_baseline_ave] + results_per_lang_ave_.tolist()))
          
          
          
#    print('{}\t'.format('Amb.')+max_markup([baseline_amb_ave] + [pos_baseline_amb_ave] + results_per_lang_amb_ave.tolist()))
#    print('{}\t{}\t'.format('-POS-unamb', '-')+max_markup([pos_baseline_amb_pos_unamb_ave] + results_per_lang_amb_pos_unamb_ave.tolist()))
#    print('{}\t{}\t'.format('-POS-amb', '-')+max_markup([pos_baseline_amb_pos_amb_ave] + results_per_lang_amb_pos_amb_ave.tolist()))
#    print('{}\t'.format('Unamb.')+max_markup([baseline_un_ave] + [pos_baseline_un_ave] + results_per_lang_un_ave.tolist()))
#    print('{}\t'.format('New')+max_markup([baseline_new_ave] + [pos_baseline_new_ave] + results_per_lang_new_ave.tolist()))
#    print('{}\t'.format('Total')+max_markup([baseline_total_ave] + [pos_baseline_total_ave] + results_per_lang_total_ave.tolist()))
print

