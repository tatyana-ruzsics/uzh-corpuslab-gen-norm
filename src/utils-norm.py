# Preprocess normalization OpenSubtitles data files
import sys
import codecs
import string

file_in = sys.argv[1]
file_out = sys.argv[2]

f_out = codecs.open(file_out, 'w', 'utf8')
with codecs.open(file_in, 'r', 'utf8') as f_in:
    for i,line in enumerate(f_in):
        splt = line.strip().split()
        new_splt = [token.lower() for token in splt if not token in string.punctuation]
        if len(new_splt) > 0:
            f_out.write(u' '.join(new_splt)+'\n')


