#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:27:35 2019

@author: sucheta
"""
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from scipy.stats.stats import pearsonr
from numpy import corrcoef
from sklearn.gaussian_process import GaussianProcessRegressor

#np.set_printoptions(threshold=np.nan)

#def stand(df):
#    name = df.columns
#    x=df.values
#    ind=df.index
#    std_scaler = preprocessing.StandardScaler()
#    x_std=std_scaler.fit_transform(x)
#    df_std = pd.DataFrame(x_std, columns=name,index=ind)
#    df = df_std.copy()
#    return (df)

#----------------  Loading PCA values ---------------------------------------------
data = pd.read_csv('/home/sucheta/Documents/ML-image/data_files/all-features.csv')


#X = data[['volume fraction','boundary fraction','objects','aspect ratio','effarea','effperi','effcompact',
#             'effrad','melting point',
#                  'boiling point','electronegativity','electron affinity','valency','1st ionization',
#                   'radius calc','Bulk Modulus','density','Thermal conductivity','Specific Heat','ageing temp','ageing time(hrs)',
#                   'Al','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zr','Nb','Mo','Hf','Ta','W',
#                   'Re','C','B','Si','P','S','v']]
#---------------------Image + compositional = ( 31 features)---------------------------#
#X=data[['Al','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zr','Nb','Mo','Hf','Ta','W',
#                   'Re','C','B','Si','P','S','v']]


#X=data[['objects', 'effarea', 'effrad', 'melting point',
#       'boiling point', '1st ionization', 'Bulk Modulus', 'density',
#       'Specific Heat', 'ageing temp', 'ageing time(hrs)', 'V', 'Fe', 'Ni','v']]
X = data[['effarea', 'objects', 'melting point', 'boiling point',
       '1st ionization', 'density', 'Specific Heat',
       'ageing temp', 'ageing time(hrs)', 'V', 'Fe', 'Ni','v']]


## =============================================================================
##                       Correlation between Y and X 
## =============================================================================
#
#
#df = data               ##------for all features
df=X                 ##----for selective features LASSO
d = pd.DataFrame(df)
#corr = d.corr()
#mm = df.columns
#print (mm)
col = 'v' 
correlation_matrix = df.corr()
correlation_type = correlation_matrix[col].copy()
abs_correlation_type = correlation_type.apply(lambda x:abs(x))
desc_corr_values = correlation_type.sort_values(ascending=False)
y_values = list(desc_corr_values.values)[1:]
#y_val1=[ j for j in y_values if j >= 0.0 ]  # for positive axis

y_val1=[ j for j in y_values]
x_val1=range(0,len(y_val1))
x_values = range(0,len(y_val1))
xlabels = list(desc_corr_values.keys())[1:len(x_val1)+1]
fig, ax = plt.subplots(figsize=(8,8))
ax.bar(x_val1, y_val1)
ax.set_title('Features'.format(df), fontsize=20)
ax.set_ylabel('Pearson correlation coefficient', fontsize=20)
plt.xticks(x_val1, xlabels, rotation='vertical')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.grid(False)
plt.savefig('/home/sucheta/Documents/ML-image/new_calc_110/correlation.eps',dpi=100,bbox_inches="tight")
plt.show()
y_val1=np.array(y_val1)
ylen=y_val1.shape[0]
y_val1=y_val1.reshape((ylen,1))
xlabels=np.array(xlabels)
xlen=xlabels.shape[0]
xlabels=xlabels.reshape((xlen,1))
array1=np.concatenate((xlabels,y_val1),axis=1)
print('--------------- THE COMPONENTS ARE ----------------')
print([str(x) for x in xlabels])
