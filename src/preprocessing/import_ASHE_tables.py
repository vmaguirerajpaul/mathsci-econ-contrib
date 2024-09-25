#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Import and pre-process data from ONS-ASHE tables across many years
"""

__author__ = 'Vinesh Maguire Rajpaul'
__email__ = 'vr325@cantab.ac.uk'
__status__ = 'Development'

import pickle
import pandas as pd
import numpy as np

print('Importing and pickling ONS-ASHE tables...', end='')

# Full-time only or all (full-time + part-time) workers
FTPT_all = ['FT', 'FTPT']
# Hourly salary tables or gross annual salary tables
HRAN_all = ['hourly', 'annual']

# Define years (and corresponding SOC systems) for CSV tables used
# Suffix p/r: provisional vs revised ONS data
CSV_years = ['2010r', '2011p', '2011r',
             '2021p', '2021r', '2023p']

# ASHE column headings: whether SOC2000, SOC2010, or SOC2020 used
SOC_frameworks = ['SOC2000_code', 'SOC2000_code', 'SOC2010_code',
                  'SOC2010_code', 'SOC2020_code', 'SOC2020_code']

# Define all percentile levels in ASHE tables
PCTL_LEVELS_ALL = ['10', '20', '30', '40', '50', '60', '70', '80', '90']


def convert_xlsx_to_csv():
    """
    Convert ASHE .xlsx files (viz. Table 14 across years) to CSV via
    pandas. Could convert directly to DataFrame then pickle, sans
    converting to CSV first; doing this way, here, for compatibility w
    with older code
    """

    ALL_EXCEL_FNS = ['2010r', '2011p', '2011r', '2021p', '2021r', '2023p']

    for excel_fn in ALL_EXCEL_FNS:

        excel_path = '../data/raw/ASHE_Table14/xlsx/' + excel_fn + '.xlsx'
        all_sheets = pd.read_excel(excel_path, sheet_name=None)
        sheets = all_sheets.keys()
        for sheet_name in sheets:
            sheet = pd.read_excel(excel_path, sheet_name=sheet_name)
            sheet.to_csv('../data/raw/ASHE_Table14/csv/%s_%s.csv' % (
                excel_fn, sheet_name), index=False)
    return None


def ASHE_DataFrame_to_dict(ASHE_DF, SOC_code):
    """
    Extract variables of interest from ASHE DataFrame (as read directly
    from a CSV file) and store in a dictionary
    """

    # Total no of people in a given occupation
    no_jobs = np.zeros(len(ASHE_DF['No_jobs'].values),)
    for i, record in enumerate(ASHE_DF['No_jobs'].values):
        try:
            no_jobs[i] = np.double(record) # thousands
        except:
            no_jobs[i] = np.nan  # job numbers not available

    # Mean salaries
    mean = np.zeros(len(ASHE_DF['Mean'].values),)
    for i, record in enumerate(ASHE_DF['Mean'].values):
        try:
            mean[i] = np.double(record)
        except:
            mean[i] = np.nan

    # Median salaries
    median = np.zeros(len(ASHE_DF['Median'].values),)
    for i, record in enumerate(ASHE_DF['Median'].values):
        try:
            median[i] = np.double(record)
        except:
            median[i] = np.nan
    n = len(ASHE_DF['Median'].values)

    # Salary percentile scores
    pctl_lvls = np.array([[]]*n + [[1]], dtype=object)[:-1]
    pctl_salaries = np.array([[]]*n + [[1]], dtype=object)[:-1]

    for i, code in enumerate(ASHE_DF['Code']):
        pctl_lvl = []
        pctl_salary = []

        for j, lvl in enumerate(PCTL_LEVELS_ALL):
            try:
                if (not lvl == '50'):
                    salary = np.double(ASHE_DF[lvl].values[i])
                # 50th percentile stored as 'Median' in ASHE table
                else:
                    salary = np.double(ASHE_DF['Median'].values[i])
            # Not all percentile scores available for every job
            except (ValueError, AttributeError):
                pctl_lvl.append(np.nan)
                pctl_salary.append(np.nan)
            else:
                pctl_lvl.append(lvl)
                pctl_salary.append(salary)

        pctl_lvls[i] = pctl_lvl
        pctl_salaries[i] = pctl_salary

    ASHE_dict = {SOC_coding: ASHE_DF['Code'].values,
                 'description': ASHE_DF['Description'].values,
                 'no_jobs': no_jobs,
                 'mean_salary': mean,
                 'median_salary': median,
                 'pctl_lvls': pctl_lvls, 'pctl_salaries': pctl_salaries
                 }

    return ASHE_dict

# Convert all ASHE table 14 XLS files to to CSV files
CSV_root = '../data/raw/ASHE_Table14/csv/'
convert_xlsx_to_csv()

# Dictionary to store all ASHE Table 14 dictionaries
ASHE_dicts_all = {}

# For a given year, iterate over all combinations: full-time vs
# part-time, and hourly vs annual data
for ftpt in FTPT_all:
    for hran in HRAN_all:

        dicts_per_year = {}

        for CSV_year, SOC_coding in zip(CSV_years, SOC_frameworks):
            CSV_name = CSV_year + '_' + hran + '_' + ftpt + '.csv'
            # Read in ASHE table; remove thousands separators (comma)
            ASHE_DataFrame = pd.read_csv(CSV_root + CSV_name,
                thousands=',')
            current_dict = ASHE_DataFrame_to_dict(
                ASHE_DataFrame, SOC_coding)
            ASHE_dicts_all[CSV_name[0:-4]] = current_dict

# Write dictionary of ASHE dictionaries to disk
with open('../data/processed/ASHE_dicts_all.pickle', 'wb') as handle:
    pickle.dump(ASHE_dicts_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('done.')
