'''
This script evaluates the LSTM model trained WITH RESOURCES
It must encode resources the same way as training!

Author: Modified from Niek Tax's original script
'''

from __future__ import division
from tensorflow.keras.models import load_model
import csv
import copy
import numpy as np
import distance
import sys

# Python 3 compatibility
if sys.version_info[0] >= 3:
    unicode = str
    unichr = chr
    try:
        from itertools import izip
    except ImportError:
        izip = zip

from jellyfish._jellyfish import damerau_levenshtein_distance
import unicodecsv
from sklearn import metrics
from math import sqrt
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

eventlog = "bpic2012_w_resources.csv"  # Dataset WITH resources!

########################################################################################
# FIRST PASS: Read data and build resource encoding
########################################################################################

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161

lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
timeseqs = []
timeseqs2 = []
resources_per_case = []  # NEW: Track resources
times = []
times2 = []
current_case_resources = []  # NEW
numlines = 0
casestarttime = None
lasteventtime = None

for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    resource = row[3] if len(row) > 3 else 'UNKNOWN'  # NEW: Get resource
    
    if row[0]!=lastcase:
        caseids.append(row[0])
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            resources_per_case.append(current_case_resources)  # NEW
        line = ''
        times = []
        times2 = []
        current_case_resources = []  # NEW
        numlines+=1
    
    line+=unichr(int(row[1])+ascii_offset)
    current_case_resources.append(resource)  # NEW
    
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
resources_per_case.append(current_case_resources)  # NEW
numlines+=1

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))
divisor3 = np.mean(list(map(lambda x: np.mean(list(map(lambda y: x[len(x)-1]-y, x))), timeseqs2)))
print('divisor3: {}'.format(divisor3))

# NEW: Build resource encoding (must match training exactly!)
all_resources = [r for case_res in resources_per_case for r in case_res]
unique_resources = sorted(list(set(all_resources)))
resource_indices = dict((r, i) for i, r in enumerate(unique_resources))
num_resources = len(unique_resources)
print('total resources: {}'.format(num_resources))

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_c = caseids[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_r = resources_per_case[:elems_per_fold]  # NEW

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_c = caseids[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
fold2_r = resources_per_case[elems_per_fold:2*elems_per_fold]  # NEW

lines = fold1 + fold2
caseids = fold1_c + fold2_c
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_r = fold1_r + fold2_r  # NEW

step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x+'!',lines))
maxlen = max(map(lambda x: len(x),lines))

chars = list(map(lambda x : set(x),lines))
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(indices_char)

########################################################################################
# SECOND PASS: Read test data with resources
########################################################################################

lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []
resources_per_case = []  # NEW
times = []
times2 = []
times3 = []
current_case_resources = []  # NEW
numlines = 0
casestarttime = None
lasteventtime = None

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers

for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    resource = row[3] if len(row) > 3 else 'UNKNOWN'  # NEW
    
    if row[0]!=lastcase:
        caseids.append(row[0])
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
            resources_per_case.append(current_case_resources)  # NEW
        line = ''
        times = []
        times2 = []
        times3 = []
        current_case_resources = []  # NEW
        numlines+=1
    
    line+=unichr(int(row[1])+ascii_offset)
    current_case_resources.append(resource)  # NEW
    
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    times3.append(datetime.fromtimestamp(time.mktime(t)))
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
resources_per_case.append(current_case_resources)  # NEW
numlines+=1

fold3 = lines[2*elems_per_fold:]
fold3_c = caseids[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
fold3_r = resources_per_case[2*elems_per_fold:]  # NEW

lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
lines_r = fold3_r  # NEW

# set parameters
predict_size = maxlen

# Load model with resources
model = load_model('output_files/models/model_resources_29-1.53.h5', compile=False)

# Define helper functions with RESOURCE encoding
def encode(sentence, times, times3, resources, maxlen=maxlen):
    num_features = len(chars) + 5 + num_resources  # Add resources!
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    times2 = np.cumsum(times)
    
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t]-midnight
        multiset_abstraction = Counter(sentence[:t+1])
        
        # Activity encoding
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        
        # Time features
        X[0, t+leftpad, len(chars)] = t+1
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = timesincemidnight.seconds/86400
        X[0, t+leftpad, len(chars)+4] = times3[t].weekday()/7
        
        # NEW: Resource encoding (one-hot)
        resource = resources[t]
        if resource in resource_indices:
            resource_idx = resource_indices[resource]
            X[0, t+leftpad, len(chars)+5+resource_idx] = 1
    
    return X

def getSymbol(predictions):
    maxPrediction = 0
    symbol = ''
    i = 0;
    for prediction in predictions:
        if(prediction>=maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []

# Make predictions
print('Making predictions with RESOURCE features...')
with open('output_files/results/suffix_and_remaining_time_RESOURCES_%s' % eventlog, 'w', newline='', encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["CaseID", "Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times", "Predicted times", "RMSE", "MAE"])
    
    for prefix_size in range(2,maxlen):
        print(prefix_size)
        for line, caseid, times, times2, times3, resources in izip(lines, caseids, lines_t, lines_t2, lines_t3, lines_r):
            times.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            cropped_resources = resources[:prefix_size]  # NEW
            
            if len(times2)<prefix_size:
                continue
            
            ground_truth = ''.join(line[prefix_size:prefix_size+predict_size])
            ground_truth_t = times2[prefix_size-1]
            case_end_time = times2[len(times2)-1]
            ground_truth_t = case_end_time-ground_truth_t
            predicted = ''
            total_predicted_time = 0
            
            for i in range(predict_size):
                enc = encode(cropped_line, cropped_times, cropped_times3, cropped_resources)  # NEW: Pass resources!
                y = model.predict(enc, verbose=0)
                
                y_char = y[0][0] 
                y_t = y[1][0][0]
                prediction = getSymbol(y_char)
                cropped_line += prediction
                
                if y_t<0:
                    y_t=0
                cropped_times.append(y_t)
                
                # NEW: For predicted events, use last known resource (simple heuristic)
                if len(cropped_resources) > 0:
                    cropped_resources.append(cropped_resources[-1])
                else:
                    cropped_resources.append('UNKNOWN')
                
                if prediction == '!':
                    one_ahead_pred.append(total_predicted_time)
                    one_ahead_gt.append(ground_truth_t)
                    print('! predicted, end case')
                    break
                
                y_t = y_t * divisor3
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                total_predicted_time = total_predicted_time + y_t
                predicted += prediction
            
            output = []
            if len(ground_truth)>0:
                output.append(caseid)
                output.append(prefix_size)
                output.append(unicode(ground_truth))
                output.append(unicode(predicted))
                output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                dls = 1 - (damerau_levenshtein_distance(unicode(predicted), unicode(ground_truth)) / max(len(predicted),len(ground_truth)))
                if dls<0:
                    dls=0
                output.append(dls)
                output.append(1 - distance.jaccard(predicted, ground_truth))
                output.append(ground_truth_t)
                output.append(total_predicted_time)
                output.append('')
                output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                spamwriter.writerow(output)

print('\n✅ Evaluation complete! Results saved to: suffix_and_remaining_time_RESOURCES_%s' % eventlog)