#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Import ONS data on self-employed (SE) vs employee jobs
"""

__author__ = 'Vinesh Maguire Rajpaul'
__email__ = 'vr325@cantab.ac.uk'
__status__ = 'Development'

import pickle
import numpy as np
import pandas as pd

print('Importing ONS self-employment job data..', end='')

excel_path = '../data/raw/self-employment/SE_jobs04sep2024.xls'

SE = pd.read_excel(excel_path, sheet_name='1. UK totals',
                          header=5)
SE.columns = SE.columns.astype(str)

SIC_2d_all = [str(code+1).rjust(2,'0') for code in range(99)]

SE_jobs_vs_SIC = {}

for GVA_year in ['2011','2023']: # compute for 2011, 2023

    quarters_all = SE['SIC 2007 division'].values
    month_labels = ['Mar', 'Jun', 'Sep', 'Dec']

    GVA_year_label = GVA_year[-2::]
    # Identify all four quarters in relevant GVA_year
    quarters_to_use = [f'{q} {GVA_year_label} ' for q in month_labels]
    # Column indices for above quarters in relevant GVA year
    quarters_ixs = [np.where(quarters_all==q)[0][0] for q in quarters_to_use]

    # Dict to store self-employment jobs vs SIC2007 code
    SE_jobs = {}

    for SIC_2d in SIC_2d_all:
        if not SIC_2d in SE.columns.values:
            SIC_2d_jobs = np.nan
        else:
            SIC_2d_jobs = SE[SIC_2d].iloc[quarters_ixs].sum()

        SE_jobs[SIC_2d] = SIC_2d_jobs
    SE_jobs_vs_SIC[GVA_year] = SE_jobs

# Pickle SE job data dictionary
with open('../data/processed/SE_jobs_vs_SIC.pickle', 'wb') as handle:
    pickle.dump(SE_jobs_vs_SIC, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('done.')
