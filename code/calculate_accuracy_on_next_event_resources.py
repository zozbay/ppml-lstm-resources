'''
This script calculates accuracy for the RESOURCES model
Takes input from evaluate_suffix_and_remaining_time_resources.py

Author: Modified from Niek Tax's original
'''

from __future__ import division
import csv

eventlog = "bpic2012_w_resources.csv"
csvfile = open('output_files/results/suffix_and_remaining_time_RESOURCES_%s' % eventlog, 'r', encoding='utf-8')
r = csv.reader(csvfile)
next(r)  # header

vals = dict()
for row in r:
    l = list()
    if row[0] in vals.keys():
        l = vals.get(row[0])
    if len(row[2])==0 and len(row[3])==0:
        l.append(1)
    elif len(row[2])==0 and len(row[3])>0:
        l.append(0)
    elif len(row[2])>0 and len(row[3])==0:
        l.append(0)
    else:
        l.append(int(row[2][0]==row[3][0]))
    vals[row[0]] = l
    
l2 = list()
for k in vals.keys():
    l2.extend(vals[k])
    res = sum(vals[k])/len(vals[k])
    print('{}: {}'.format(k, res))

print('total: {}'.format(sum(l2)/len(l2)))