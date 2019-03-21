
import glob
import sys
import re
import numpy as np

PREFIX = sys.argv[1] # eng or ger

results = {}
stats = {}

model = 'nmt'
stats['seen'] = []
stats['new'] = []
stats['new morphemes'] = []
stats['new combinations'] = []
for fold in range(10):
    result_path = glob.glob('../results/{}/ensemble/{}/{}/test.eval.det'.format(PREFIX,model,fold))
    with open(result_path[0]) as f:
        for line in f:
            m_seen_s = re.search('(.*)unique\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_new_s = re.search('(.*)new\ssource\sword\stokens:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_morf_s = re.search('(.*)new\ssource\sword\stokens\s-\snew\starget\ssegments:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            m_comb_s = re.search('(.*)new\ssource\sword\stokens\s-\snew\scombination:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
            # categories statistics
            if m_seen_s:
                stats['seen'].append(float(m_seen_s.group(5)))
            if m_new_s:
                stats['new'].append(float(m_new_s.group(5)))
            if m_morf_s:
                stats['new morphemes'].append(float(m_morf_s.group(5)))
            if m_comb_s:
                stats['new combinations'].append(float(m_comb_s.group(5)))
print()
print('10-fold statistics:')
for category in stats.keys():
    values = np.array(stats[category])
#    print(values)
    mean = np.mean(values)
    std = np.std(values)
    print('{}: {:.2f} ({:.2f})'.format(category, mean, std ))
print()
print('5-fold statistics:')
for category in stats.keys():
    values = np.array(stats[category][:5])
    #print(values)
    mean = np.mean(values)
    std = np.std(values)
    print('{}: {:.2f} ({:.2f})'.format(category, mean, std ))
print()

for model in ['nmt','we']:
    print()
    print('Model {}:'.format(model))
    results[model] = {}
    results[model]['total'] = []
    results[model]['new'] = []
    results[model]['new morphemes'] = []
    results[model]['new combinations'] = []
    for fold in range(10):
        result_path = glob.glob('../results/{}/ensemble/{}/{}/test.eval.det'.format(PREFIX,model,fold))
#        print(result_path[0])
        with open(result_path[0]) as f:
            for line in f:
                #print(line)
                m_total = re.search('(\s+)Number\sof\scorrect\spredictions\stotal:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_new = re.search('(\s+)-\snew:(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_morf = re.search('(\s+)-\snew\s\(new\smorphemes\):(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                m_comb = re.search('(\s+)-\snew\s\(new\scombination\):(\s+)([\d]+)(\s+)([\d\.]+)%$', line)
                
                # preformance
                if m_total:
                    results[model]['total'].append(float(m_total.group(5)))
                if m_new:
                    results[model]['new'].append(float(m_new.group(5)))
                if m_morf:
                    results[model]['new morphemes'].append(float(m_morf.group(5)))
                if m_comb:
                    results[model]['new combinations'].append(float(m_comb.group(5)))

    print('10-fold evaluation:')
    for category in results[model].keys():
        values = np.array(results[model][category])
        #print(values)
        mean = np.mean(values)/100
        std = np.std(values)/100
        print('{}: {:.4f} ({:.4f})  [{:.4f}]'.format(category, mean, std, 1-mean ))
    print('5-fold evaluation:')
    for category in results[model].keys():
        values = np.array(results[model][category][:5])
        #print(values)
        mean = np.mean(values)/100
        std = np.std(values)/100
        print('{}: {:.4f} ({:.4f})  [{:.4f}]'.format(category, mean, std, 1-mean ))




