'''
This script trains an LSTM model with RESOURCE INFORMATION
Reading from bpic2012_w_resources.csv which includes resource column

This implements the future work suggested by Tax et al. (2017):
"extend feature vectors with additional case and event attributes (e.g. resources)"

Author: Modified from Niek Tax's original script
'''

from __future__ import print_function, division
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Input, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from collections import Counter
import unicodecsv
import numpy as np
import random
import sys

# Python 2/3 compatibility
if sys.version_info[0] >= 3:
    unicode = str
    unichr = chr
    try:
        from itertools import izip
    except ImportError:
        izip = zip

import os
import copy
import csv
import time
from datetime import datetime
from math import log

eventlog = "bpic2012_w_resources.csv"  # Dataset WITH resources!

########################################################################################
# READ DATA - First pass to get basic sequences
########################################################################################

lines = []
timeseqs = []
timeseqs2 = []
resources_per_case = []  # NEW: Track resources

lastcase = ''
line = ''
firstLine = True
times = []
times2 = []
current_case_resources = []  # NEW: Resources for current case
numlines = 0
casestarttime = None
lasteventtime = None

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161

for row in spamreader:  # CaseID, ActivityID, CompleteTimestamp, Resource
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    resource = row[3] if len(row) > 3 else 'UNKNOWN'  # NEW: Get resource
    
    if row[0] != lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            resources_per_case.append(current_case_resources)  # NEW: Save resources
        line = ''
        times = []
        times2 = []
        current_case_resources = []  # NEW: Reset resources
        numlines += 1
    
    line += unichr(int(row[1]) + ascii_offset)
    current_case_resources.append(resource)  # NEW: Track resource
    
    timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    lasteventtime = t
    firstLine = False

# Add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
resources_per_case.append(current_case_resources)  # NEW
numlines += 1

########################################

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))

# NEW: Create resource encoding
all_resources = [r for case_res in resources_per_case for r in case_res]
unique_resources = sorted(list(set(all_resources)))
resource_indices = dict((r, i) for i, r in enumerate(unique_resources))
num_resources = len(unique_resources)
print('total resources: {}'.format(num_resources))
print('resource mapping sample: {}'.format(dict(list(resource_indices.items())[:10])))

#########################################################################################################
# Separate training data into 3 parts
#########################################################################################################

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_r = resources_per_case[:elems_per_fold]  # NEW

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
fold2_r = resources_per_case[elems_per_fold:2*elems_per_fold]  # NEW

fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_r = resources_per_case[2*elems_per_fold:]  # NEW

# Use fold1 + fold2 for training
lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_r = fold1_r + fold2_r  # NEW

step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x+'!', lines))
maxlen = max(map(lambda x: len(x), lines))

chars = list(map(lambda x: set(x), lines))
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
# READ DATA - Second pass with all time features AND resources
########################################################################################

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)
lastcase = ''
line = ''
firstLine = True
lines = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []
timeseqs4 = []
resources_per_case = []  # NEW
times = []
times2 = []
times3 = []
times4 = []
current_case_resources = []  # NEW
numlines = 0
casestarttime = None
lasteventtime = None

for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    resource = row[3] if len(row) > 3 else 'UNKNOWN'  # NEW
    
    if row[0] != lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
            timeseqs4.append(times4)
            resources_per_case.append(current_case_resources)  # NEW
        line = ''
        times = []
        times2 = []
        times3 = []
        times4 = []
        current_case_resources = []  # NEW
        numlines += 1
    
    line += unichr(int(row[1]) + ascii_offset)
    current_case_resources.append(resource)  # NEW
    
    timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    timediff3 = timesincemidnight.seconds
    timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()
    times.append(timediff)
    times2.append(timediff2)
    times3.append(timediff3)
    times4.append(timediff4)
    lasteventtime = t
    firstLine = False

# Add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
timeseqs4.append(times4)
resources_per_case.append(current_case_resources)  # NEW
numlines += 1

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_t3 = timeseqs3[:elems_per_fold]
fold1_t4 = timeseqs4[:elems_per_fold]
fold1_r = resources_per_case[:elems_per_fold]  # NEW

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
fold2_t3 = timeseqs3[elems_per_fold:2*elems_per_fold]
fold2_t4 = timeseqs4[elems_per_fold:2*elems_per_fold]
fold2_r = resources_per_case[elems_per_fold:2*elems_per_fold]  # NEW

fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
fold3_t4 = timeseqs4[2*elems_per_fold:]
fold3_r = resources_per_case[2*elems_per_fold:]  # NEW

lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4
lines_r = fold1_r + fold2_r  # NEW

step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x+'!', lines))

sentences_t = []
sentences_t2 = []
sentences_t3 = []
sentences_t4 = []
sentences_r = []  # NEW: Resource sequences
next_chars_t = []
next_chars_t2 = []
next_chars_t3 = []
next_chars_t4 = []

for line, line_t, line_t2, line_t3, line_t4, line_r in izip(lines, lines_t, lines_t2, lines_t3, lines_t4, lines_r):
    for i in range(0, len(line), step):
        if i == 0:
            continue
        
        sentences.append(line[0:i])
        sentences_t.append(line_t[0:i])
        sentences_t2.append(line_t2[0:i])
        sentences_t3.append(line_t3[0:i])
        sentences_t4.append(line_t4[0:i])
        sentences_r.append(line_r[0:i])  # NEW: Resource sequence
        next_chars.append(line[i])
        
        if i == len(line)-1:
            next_chars_t.append(0)
            next_chars_t2.append(0)
            next_chars_t3.append(0)
            next_chars_t4.append(0)
        else:
            next_chars_t.append(line_t[i])
            next_chars_t2.append(line_t2[i])
            next_chars_t3.append(line_t3[i])
            next_chars_t4.append(line_t4[i])

print('nb sequences:', len(sentences))

print('Vectorization...')
# NEW: Add resource features to feature count
num_features = len(chars) + 5 + num_resources  # activities + time + RESOURCES!
print('num features: {} (activities: {}, time: 5, resources: {})'.format(num_features, len(chars), num_resources))

X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
y_t = np.zeros((len(sentences)), dtype=np.float32)

for i, sentence in enumerate(sentences):
    leftpad = maxlen - len(sentence)
    next_t = next_chars_t[i]
    sentence_t = sentences_t[i]
    sentence_t2 = sentences_t2[i]
    sentence_t3 = sentences_t3[i]
    sentence_t4 = sentences_t4[i]
    sentence_r = sentences_r[i]  # NEW: Resources for this sentence
    
    for t, char in enumerate(sentence):
        multiset_abstraction = Counter(sentence[:t+1])
        
        # Activity encoding
        for c in chars:
            if c == char:
                X[i, t+leftpad, char_indices[c]] = 1
        
        # Time features
        X[i, t+leftpad, len(chars)] = t+1
        X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
        X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
        X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/86400
        X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/7
        
        # NEW: Resource encoding (one-hot)
        resource = sentence_r[t]
        if resource in resource_indices:
            resource_idx = resource_indices[resource]
            X[i, t+leftpad, len(chars)+5+resource_idx] = 1
    
    # Target outputs
    for c in target_chars:
        if c == next_chars[i]:
            y_a[i, target_char_indices[c]] = 1 - softness
        else:
            y_a[i, target_char_indices[c]] = softness/(len(target_chars)-1)
    
    y_t[i] = next_t/divisor
    np.set_printoptions(threshold=sys.maxsize)

# Build model
print('Build LSTM model with RESOURCE features...')
main_input = Input(shape=(maxlen, num_features), name='main_input')
l1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input)
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

model = Model(inputs=[main_input], outputs=[act_output, time_output])

opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)

model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint('output_files/models/model_resources_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

print('Starting training with RESOURCES...')
model.fit(X, {'act_output':y_a, 'time_output':y_t}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, epochs=500)

print('\n✅ Training complete! Models saved with prefix: model_resources_')