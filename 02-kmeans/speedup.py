#!/bin/python3

import pandas as pd
import numpy as np
import sys
from scipy import stats

if len(sys.argv) < 2:
  print(f'usage: ./speedup.py <data_file>.csv')

df = pd.read_csv(sys.argv[1])

type_values = ['serial', 'tbb']
type_conditions = [(df['type'].str.contains('serial') == True), (df['type'].str.contains('serial') == False)]
df['type'] = np.select(type_conditions, type_values)
speedupdf = df.groupby("run")\
  .apply(lambda row: row['time'].shift(1) / row['time'])\
  .dropna()\
  .reset_index(drop=True)

print(stats.hmean(speedupdf))