import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('test_normalized.csv')

df = df.drop(['Unnamed: 0', 'ID'], axis=1)

props = df.columns

print (props)

normalizer=np.load('normalizer.npy',allow_pickle=True)

normalizer = normalizer.item()

results = {}

f=open('test_results.csv')
length = len(f.readlines())
f.close()
f=open('test_results.csv')
for i in range(length):
    id,p,true,pred=f.readline().split(',')
    prop = props[int(p)]
    if prop not in results.keys():
        results[prop] = {'true':[float(true)],'pred':[float(pred)]}
    else:
        results[prop]['true'].append(float(true))
        results[prop]['pred'].append(float(pred))

for prop in results.keys():
    print (prop, len(results[prop]['true']), r2_score(results[prop]['true'], results[prop]['pred']),
            mean_squared_error(results[prop]['true'], results[prop]['pred'])*(normalizer[prop]['p90']-normalizer[prop]['p10']))
