import pandas as pd
import numpy as np
import os

os.system('rm id_prop.csv')

df = pd.read_csv('test_normalized.csv')

df = df.drop(['Unnamed: 0','debye_temperature'],axis=1)

for index, row in df.iterrows():
    for i in range(len(df.columns)):
        col = df.columns[i]
        entry = row[col]
        if entry == 'no':
            continue
        if col == 'ID':
            mpid = entry
            continue
        if abs(float(entry)) > 2:
            continue
        if np.isnan(float(entry)):
            continue
        print ('%s,%d,%f'%(mpid, i, float(entry)), file=open('id_prop.csv','a'))

