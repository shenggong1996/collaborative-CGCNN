import pandas as pd
import random

df = pd.read_csv('mp_selected_prop.csv')

df = df.drop(['Unnamed: 0','elastic_anisotropy', 'average_linear_thermal_expansion', 'trans_v', 'long_v'],axis=1)

excluded = []

#print (df[['eij_max']])

#for col in df.columns:
#    if df.var

print (df.head())

train = []

test = []

n_total = 0

for index, row in df.iterrows():
    row_train = []
    row_test = []
    if float(row['formation_energy_per_atom']) > 0.1:
        continue
    for col in df.columns:
        entry = row[col]
        if col == 'ID':
            n_total += 1
            row_train.append(entry)
            row_test.append(entry)
            continue
        if entry != 'no':
            n_total += 1
#            if col == 'eij_max':
#                print (entry)
            p = random.random()
            if p <= 0.2:
                row_train.append('no')
                row_test.append(entry)
            else:
                row_train.append(entry)
                row_test.append('no')
        else:
            row_train.append('no')
            row_test.append('no')
    train.append(row_train)
    test.append(row_test)

print (n_total)

df_train = pd.DataFrame(data=train, columns=df.columns)
df_test = pd.DataFrame(data=test, columns=df.columns)

#print (df_train)
#print (df_test)

df_train.to_csv('training_unnormalized.csv')
df_test.to_csv('test_unnormalized.csv')

normalizer = {}

for index, row in df_train.iterrows():
    for col in df_train.columns:
        if col == 'ID' or row[col] == 'no' or row[col] != row[col]:
            continue
#        if col == 'n_index' and row[col] != 'no':
#            print (row[col])
        if col not in normalizer.keys():
            normalizer[col] = [float(row[col])]
        else:
            normalizer[col].append(float(row[col]))

print (normalizer['n_index'])

import numpy as np
for prop in normalizer.keys():
    median = np.percentile(normalizer[prop],50)
    p10 = np.percentile(normalizer[prop],5)
    p90 = np.percentile(normalizer[prop],95)
    normalizer[prop] = {'median':median,'p10':p10,'p90':p90}

train = []

for index, row in df_train.iterrows():
    row_train = []
    for col in df_train.columns:
        if col == 'ID':
            row_train.append(row[col])
            continue
        if row[col] == 'no':
            row_train.append('no')
            continue
        median = normalizer[col]['median']
        p10 = normalizer[col]['p10']
        p90 = normalizer[col]['p90']
        row_train.append((float(row[col]) - median)/(p90-p10))
     
    train.append(row_train)

df_train = pd.DataFrame(data=train, columns=df_train.columns)

print (df_train)

df_train.to_csv('training_normalized.csv')


test = []

for index, row in df_test.iterrows():
    row_test = []
    for col in df_test.columns:
        if col == 'ID':
            row_test.append(row[col])
            continue
        if row[col] == 'no':
            row_test.append('no')
            continue
        median = normalizer[col]['median']
        p10 = normalizer[col]['p10']
        p90 = normalizer[col]['p90']
        row_test.append((float(row[col]) - median)/(p90-p10))

    test.append(row_test)

df_test = pd.DataFrame(data=test, columns=df_test.columns)

print (df_test)

df_test.to_csv('test_normalized.csv')

np.save('normalizer',normalizer)

print (normalizer)

#import json

#json.dump(normalizer, 'normalizer.json')
#        print (entry)
#        break
#    break
