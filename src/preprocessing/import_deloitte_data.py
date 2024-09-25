#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Import data from Deloitte's report
"""

__author__ = 'Vinesh Maguire Rajpaul'
__email__ = 'vr325@cantab.ac.uk'
__status__ = 'Development'

import pickle
import pandas as pd
import numpy as np

print('Importing and pickling Deloitte data...', end='')

# Deloitte table on MS occupation numbers vs SOC: read in as DataFrame
DeloitteDF = pd.read_csv('../data/raw/Deloitte_fig_6_1_3.csv')

# SOC2000 codes are stores as four digit numbers a space then a title
# In the 'Level 4 SOC code' column. Extract the codes only, into an array
SOC2000_code = np.array([i[0:4] for i in DeloitteDF['Level 4 SOC code']])

# Into another array with the same indexing, extract whether the field
# 'Include entire occupation as mathematics occupation?' is Y or N,
# encoding these as 1 and 0 respectively. This is whether to include
# entirety of occupation as MS occupation
SOC2000_entirety_flag = ([1 if i.lower() == 'y' else 0 for i in DeloitteDF[
    'Include entire occupation as mathematics occupation?']])
SOC2000_entirety_flag = np.array(SOC2000_entirety_flag)

# Into another array with the same indexing, extract whether the field
# 'Apportion needed' is Y or N, encoding these as 1 and 0 respectively.
SOC2000_apportion_flag = np.array([1 if i.lower() == 'y' else 0
                                   for i in DeloitteDF['Apportion needed?']])

# Create another array with the same indexing which maps the weighting
# for each SOC code: 0 if there is no MS job, 1 if they are all MS jobs,
# a double for proportions, or nan otherwise.
SOC2000_weight = np.zeros(len(SOC2000_code))
for i, (SOC, MS_jobs) in enumerate(zip(SOC2000_code, DeloitteDF[
        'Final number of mathematical science occupations'])):

    record = DeloitteDF['% of category included'][i]
    # No apportioning needed: 100% MS according to Deloitte
    if (record == 'na') and (MS_jobs > 1):
        SOC2000_weight[i] = 1
    # 0% MS according to Deloitte
    elif (record == 'na') and (MS_jobs == 0):
        SOC2000_weight[i] = 0
    # Cases where  total no. of occupations cannot be publicly disclosed
    elif record == '*':
        SOC2000_weight[i] = np.nan
    # Deloitte's tabulated weight; remove % and divide by 100 -> ∈ [0,1]
    else:
        SOC2000_weight[i] = np.double(record[0:-1])/100

# Deloitte's final estimate of MS workers in a given occupation
SOC2000_MSjobs = np.array(DeloitteDF[
    'Final number of mathematical science occupations'])

# Total number (employed + self-employed) workers in a given occupation
SOC2000_totaljobs = np.array([np.nan if z == '*' else np.double(z) for z in
                              DeloitteDF['Total number of jobs in SOC category']])

# Store Deloitte data in dictionary
D2010 = {
    'SOC2000_code': SOC2000_code,
    'entirety_flag': SOC2000_entirety_flag,
    'apportion_flag': SOC2000_apportion_flag,
    'weight': SOC2000_weight,
    'weight_recalc': SOC2000_MSjobs/SOC2000_totaljobs,
    'MS_jobs': SOC2000_MSjobs,
}

# Fix Deloitte typos (4% instead of 100%, twice)
D2010['weight'][D2010['SOC2000_code'] == '3568'] = 1
D2010['weight'][D2010['SOC2000_code'] == '4121'] = 1

# SOC2000 code 2131's weight of 1 (Deloitte) clearly at odds with job
# numbers; therefore replace weight with manually recalculated weight
D2010['weight'][25] = D2010['weight_recalc'][25]

# Pickle Deloitte data dictionary
with open('../data/processed/D2010_data.pickle', 'wb') as handle:
    pickle.dump(D2010, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('done.')
