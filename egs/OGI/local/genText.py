#SPAPL

import sys

text = sys.argv[1]
uttids = sys.argv[2]
outfile = sys.argv[3]

text_dict = {}
with open(text, 'r') as rf:
    line = rf.readline()
    while line:
        line = line.strip().split(' ', 1)
        text_dict[line[0].lower()] = line[1][1:-1]
        line = rf.readline()

with open(uttids, 'r') as rf, open(outfile, 'w') as wf:
    line = rf.readline()
    while line:
        line = line.strip()
        key = line[-3:-1]
        trans = text_dict[key].upper().replace(',','')
        print(line + ' ' + trans, file=wf)
        line = rf.readline()
print("Generate text done!")

