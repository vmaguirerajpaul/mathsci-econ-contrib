#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute SOC-SOC re-basing defined by ONS dual coding exercises
"""

__author__ = 'Vinesh Maguire Rajpaul'
__email__ = 'vr325@cantab.ac.uk'
__status__ = 'Development'

import pickle
import pandas as pd
import numpy as np

print('Computing ONS SOC re-basing relationships...', end='')

# --- SOC2000 to SOC2010 ---

# ref: https://tinyurl.com/mrx9edw9
# NB: frequencies tabulated (% of no of people), where total people is
# different for different occupations & genders. So %s don't add to 100.

# Load ONS relationship table
X = pd.read_excel('../data/raw/ONS_SOC2000_SOC2010/Table 1b.xls',
                  header=None, sheet_name='Unit')
X = X[2::].reset_index()  # skip header lines etc

# Load ONS sample size table
Y = pd.read_excel('../data/raw/ONS_SOC2000_SOC2010/Table 2b.xls',
                  header=None, sheet_name='Unit')
Y = Y[2::].reset_index()  # skip header lines etc

# Filter: 4-digit SOC2000 codes only (strings)
SOC2000 = [(j[0:4], i) for i, j in enumerate(
    X[0]) if (isinstance(j, str) and (j != '_') and j[0] != '*')
           ]

SOC2000_codes = np.array([i[0] for i in SOC2000])

# Indices where each input 4-digit SOC2000 code is listed
SOC2000_ixs = np.array([i[1] for i in SOC2000])

mapping_2000_2010 = {}

for i, SOC2000_ix in enumerate(SOC2000_ixs):

    # Figure out where list of mapped SOC2010 codes starts and ends
    start_ix = SOC2000_ixs[i]+1
    stop_ix = (len(SOC2000) if i == len(SOC2000_ixs)-1
               else SOC2000_ixs[i+1])
    n_mapped = stop_ix - start_ix

    # All SOC2010 codes to which a given SOC2000 code maps
    mapped_codes = X[3][start_ix:stop_ix].values
    mapped_codes = np.array([str(c) for c in mapped_codes])
    mapped_wghts = [[]]*n_mapped

    # Use sample size & frequency tables to compute actual no. of people
    # mapped to given SOC2010 codes, and thus compute weights
    for j, mapped_code in enumerate(mapped_codes, 0):
        mapped_samples = Y[Y[0] == int(mapped_code)].values[0][2:8]
        mapped_samples = [0 if isinstance(k, str) else k for k in mapped_samples]

        mapped_wghts[j] = X[[0, 1, 2, 5, 6, 7]].values[start_ix + j]
        mapped_wghts[j] = (np.array([0 if isinstance(k, str)
                                     else k for k in mapped_wghts[j]]) * mapped_samples / 100).sum()
    # Save current code's mapping (output codes & weights) in a dictionary
    mapping_current_code = {'output_codes': mapped_codes,
                            'weights': np.array(mapped_wghts)}

    mapping_2000_2010[str(SOC2000_codes[i])] = mapping_current_code


# --- SOC2010 to SOC2020 ---
# Note that the format of the relationship tables for SOC2010 to SOC2020
# is different to (simpler than) for SOC2000 to SOC2010, so the code
# below is not identical to above. Now, frequency and sample size data
# are tabulated in a single Excel sheet.

X = pd.read_excel('../data/raw/ONS_SOC2010_SOC2020/ONS_SOC2010_SOC2020.xlsx',
                  sheet_name='Sheet1')

# Remove NaNs
X['SOC2010_code'] = [i[0:4] for i in X['SOC2010'].fillna('-----')]
X['SOC2020_code'] = [i[0:4] for i in X['SOC2020'].fillna('-----')]

# Index the start of each input SOC2010 code (4-digit)
SOC2010_ixs = np.where(X['SOC2010_code'] != '----')[0]

# Col headers: total sample size
total_sample_hdrs = ['men_lfs_base_2010', 'men_census_base_2010',
                     'women_lfs_base_2010', 'women_census_base_2010']

# Col headers: % of SOC2010 code mapped to given SOC2020 code
job_pct_hdrs = ['men_lfs_perc_2020', 'men_census_perc_2020',
                'women_lfs_perc_2020', 'women_census_perc_2020']

# Dictionary to store entire mapping
mapping_2010_2020 = {}

for i, SOC2010_ix in enumerate(SOC2010_ixs):

    # Figure out where list of mapped SOC2020 codes starts and ends
    start_ix = SOC2010_ixs[i] + 1  # skip "Total sample" lines
    stop_ix = (len(X['SOC2010']) if i == len(SOC2010_ixs)-1
               else SOC2010_ixs[i+1])
    n_mapped = stop_ix - start_ix

    # All SOC2020 codes to which a given SOC2010 code maps
    mapped_codes = np.array(X['SOC2020_code'][start_ix:stop_ix].values)
    mapped_sample = np.array([X[hdr].values[SOC2010_ix]
                              for hdr in total_sample_hdrs])

    """ Compute weights for SOC2010 code's SOC2020 mapping
    VMR note to self: multiplying the %s by the sample size produces,
    to within double precision, integers - because the %s were
    presumably computed from integer numbers (people), not vice versa
    """
    mapped_wghts = np.zeros(len(mapped_codes))
    for j, mapped_code in enumerate(mapped_codes, 1):
        mapped_wghts[j-1] = (np.array([X[hdr].values[SOC2010_ix+j]
                                       for hdr in job_pct_hdrs]*mapped_sample
                                      ).sum())

    # Save current code's mapping (output codes & weights) in dict
    mapping_current_code = {'output_codes': mapped_codes,
                            'weights': mapped_wghts}

    mapping_2010_2020[X['SOC2010_code'][SOC2010_ix]] = mapping_current_code

# Save both mappings in a single dictionary
mappings_all = {'2000_2010': mapping_2000_2010,
                '2010_2020': mapping_2010_2020}

# Pickle SOC relationship dictionary
with open('../data/processed/ONS_SOC_relationships.pickle', 'wb') as handle:
    pickle.dump(mappings_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('done.')
