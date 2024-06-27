# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:15:31 2022

"""

from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from pandas import read_excel
import seaborn as sns
A = read_excel('D:/python/New folder/Dataframe.xlsx', header=0, )
A = A.drop(['Date'], axis = 1)
B = ['A','Exchange rate', 'Sum_rain', 'Avg_Temp', 'Yield', 'CPI','Fertilize Price']
#calculation correlation coefficient and p-value between x and y
print('(Correlation , P-value)\n')
for i in range(1,7,1):    
    a = pearsonr(A.iloc[:,i], A.iloc[:,0])
    print(B[i],'correlation test')
    print(a,'\n')
