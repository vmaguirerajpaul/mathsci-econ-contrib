#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translate mathematical science workforce into GVA estimates for 2023
"""

__author__ = 'Vinesh Maguire Rajpaul'
__email__ = 'vr325@cantab.ac.uk'
__status__ = 'Development'

import pickle
import pandas as pd
import numpy as np

print('Estimating GVA contribution from MS workforce estimates...')

# short-hand pointer
pctl = np.percentile

# whether to save output to disk
SAVE_PICKLE = True

# whether to print debugging output to screen
DEBUG = False

# how much output to print on screen (= 0, 1, 2)
VERBOSE = 2

# whether to include self-employed in total workforce calculation
INCL_SELF_EMP = True

# Compute GVA estimate for 2023 (code also works for 2010, to compare my
# methodology's output with Deloitte's)
OUTPUT_YEAR = '2023'

# ONS UK GVA totals for various years [https://tinyurl.com/2meha42b]
GVA = pd.read_csv('../data/raw/GVA/GVA-series-141024.csv', header=None)
UK_GVA = {year: np.double(GVA[1][np.where(GVA[0].values == year)[0][0]])/1000
          for year in ['2010', '2011', '2022', '2023']}

# For 2010, work with 2011 job data (to avoid re-basing to SOC2000) and
# then re-scale to 2010 at end.
jobs_year = SIC_year = ('2011' if OUTPUT_YEAR == '2010' else '2023')

# Also, 2023 GVA/industry data not yet available; use 2022 as best proxy
gva_year = '2011' if OUTPUT_YEAR == '2010' else '2022'

# Self employed job data (totals and workforce ratios)
with open('../data/processed/SE_jobs_vs_SIC.pickle', 'rb') as input_file:
    SE_jobs_vs_SIC = pickle.load(input_file)[jobs_year]

SE_WF_ratios = pd.read_csv(
    f'../data/raw/self-employment/SE_WF_ratios_{jobs_year}.csv')

# Load MS job dictionaries
with open('../data/processed/MSW_all.pickle', 'rb') as input_file:
    MSW_all = pickle.load(input_file)

# Total numbers of UK workers in various years
# via https://tinyurl.com/y62x58nf [ONS]
wf_size = pd.read_csv('../data/raw/people_in_jobs.csv')
wf_size = {str(y): j for (y, j) in zip(wf_size['year'].values,
                                       wf_size['jobs'].values)}

# Import ONS SOC-SIC matrix
SOC_SIC = pd.read_csv('../data/raw/SOC-SIC-matrix.csv')
SOC_left = [SOC_SIC['Unnamed: 0'][i][1:5] for i in range(SOC_SIC.shape[0])]
SOC_right = [SOC_SIC['Unnamed: 0'][i][6:10] for i in range(SOC_SIC.shape[0])]
assert SOC_left == SOC_right

# If SOC_left == SOC_right, can consider either one of them the SOC2010 index.
SOC2010 = SOC_left

# Extract SIC07 codes from the SOC-SIC matrix
SIC07_2d, SIC07_4d = [], []
for ix, key in enumerate(SOC_SIC.keys()[1::]):
    stop_ix = key.find('.')
    start_ix = stop_ix-2
    SIC07_code = key[start_ix:stop_ix]
    SIC07_4d_code = key[start_ix:stop_ix+3]
    SIC07_2d.append(SIC07_code)
    SIC07_4d.append(SIC07_4d_code)

# Extract SIC07 sector letters from SE_WF_ratios_{year}.csv
SIC07_letter = ({SE_WF_ratios['SIC07_2d'][i]: SE_WF_ratios['Sector'][i]
                 for i in range(len(SE_WF_ratios))})
# Remove "XX" dummy code from list of 2D codes
_ = SIC07_letter.pop('XX', None)

# Column labels and indices for SOC-SIC matrix
SIC_4d_labels = {key: ix for ix, key in enumerate(SIC07_4d)}
SIC_2d_labels = np.unique(SIC07_2d)

# Dictionary to translate 4-digit SIC codes into 2-digit SIC codes
SIC_4d_2d = {k1: k2 for k1, k2 in zip(SIC07_4d, SIC07_2d)}

# Row labels and indices for SOC-SIC matrix
SOC_labels = {key: ix for ix, key in enumerate(SOC2010)}

# Create SOC-SIC numeric matrix for computing weights (2018, S2010);
# ignore first column, which isn't a SIC code
SS = np.double(SOC_SIC.values[:, 1::])

# Lists to store all GVA, productivity, and workforce estimates
GVA_all, prod_all, MS_WF_all = [], [], []

# Counter: total number of configurations run
n_config = 0

for k, v in MSW_all.items():

    curr_MSW = MSW_all[k][f'MSW_{jobs_year}_SOC2010']

    # Need to map SOC employee numbers to SIC employee numbers
    SIC_MS = np.zeros(len(SIC_4d_labels))

    SOC_MS_tot = 0  # for debugging only

    for i, SOC_code in enumerate(curr_MSW['SOC'].values):

        row = SOC_labels[SOC_code]
        row_norm = np.sum(SS[row, :])
        SIC_map_fracs = SS[row, :]/row_norm if row_norm != 0 else 0*SS[row, :]

        # NB to use SOC2010 scheme job vector, not SOC2020s
        SOC_MS_tot += curr_MSW['MS_jobs'].values[i]  # debugging only
        SIC_MS += SIC_map_fracs*curr_MSW['MS_jobs'].values[i]

    chk1 = sum(curr_MSW['MS_jobs'].values)
    chk2, chk3 = SOC_MS_tot, SIC_MS.sum()
    if DEBUG:
        print(chk1/chk2, chk1/chk3, chk2/chk3)  # should all be unity

    # Figure out fraction of employment in each SIC code that is MS
    ONS_SIC = pd.read_csv(f'../data/raw/ASHE_Table4/csv/{SIC_year}' +
                          '_FTPT_hourly.csv')

    SIC_code_2D = []
    for code in ONS_SIC['Code']:
        code = str(code)
        # Convert "1" to "01", "2" to "02", etc.
        if len(code) == 1:
            code = '0'+code
        try:
            np.double(code)
        except:
            SIC_code_2D.append('XX')  # placeholder for non-numeric codes
        else:
            SIC_code_2D.append(code)  # if code can be converetd to double

    ONS_SIC['Code_2D'] = SIC_code_2D

    # Condense 4-digit SIC_MS_2020 matrix into 2-digit array
    SIC2d_MS = {label: 0 for label in SIC_2d_labels}

    for i, code4d in enumerate(SIC_4d_labels):
        # get corresponding 2D code
        code2d = SIC_4d_2d[code4d]
        SIC2d_MS[code2d] += SIC_MS[i]

    if DEBUG:
        print('Total MS jobs mapped to SIC 2-digit codes:',
              np.sum([SIC2d_MS[k] for k in SIC2d_MS]))

    # Dictionary to store MS worker frac vs SIC code, using ONS data
    SIC2d_MS_frac = {code2d: 0 for code2d in SIC_2d_labels}

    # Factor in self-employed dilution in numerator (workforce size)

    MS_jobs_mapped_into_fracs, tot_jobs_to_redist = 0, 0

    for code in SIC2d_MS_frac:

        employee_jobs = ONS_SIC['Jobs'][ONS_SIC['Code_2D'] == code].values[0]
        employee_jobs = str(employee_jobs).replace(',', '')

        try:
            employee_jobs = np.double(employee_jobs)*1000
        except ValueError:
            if DEBUG:
                print(employee_jobs)
            employee_jobs, total_jobs = np.nan, np.nan

        if INCL_SELF_EMP:

            # Use most fine-grained SE job data, if available
            if code in SE_jobs_vs_SIC:
                SE_jobs = SE_jobs_vs_SIC[code]
            # Else use estimate SE fraction from sector-level data
            else:
                SE_WF_ratio = SE_WF_ratios['SE_frac'][SE_WF_ratios['SIC07_2d']
                                                      == code].values
                SE_jobs = employee_jobs*SE_WF_ratio/(1-SE_WF_ratio)
            total_jobs = (employee_jobs + SE_jobs)
            SIC2d_MS_frac[code] = SIC2d_MS[code]/total_jobs
        else:
            SIC2d_MS_frac[code] = SIC2d_MS[code]/employee_jobs

        # for counting how many MS jobs unmapped
        MS_jobs_mapped_into_fracs += SIC2d_MS[code]

    # Fraction of MS jobs mapped into SIC sector fractional employments
    MS_jobs_unmapped_into_fracs = np.sum(SIC_MS)-MS_jobs_mapped_into_fracs
    # correction_factor = np.sum(SIC_MS)/MS_jobs_mapped_into_fracs

    SIC2d_MS = {k: v for (k, v) in zip(SIC2d_MS.keys(), SIC2d_MS.values())}

    # Map MS workers to industry letter codes as well
    SIC2d_MS_dict = {k: v for k, v in zip(SIC2d_MS.keys(), SIC2d_MS.values())}
    SIC_letter_MS = ({letter: 0 for letter in sorted(set(
        SIC07_letter.values()))})

    for letter in SIC_letter_MS:
        letter_MS_total = 0
        for (key, val) in zip(SIC2d_MS_dict.keys(), SIC2d_MS_dict.values()):
            if letter == SIC07_letter[key]:
                letter_MS_total += val
        SIC_letter_MS[letter] = letter_MS_total

    SIC_letter_MS_frac = {letter: (100*v/sum(SIC2d_MS_dict.values())
                                   ) for (letter, v) in zip(SIC_letter_MS.keys(), SIC_letter_MS.values())}

    letters = np.array(sorted(set(SIC07_letter.values())))
    letter_MS_frac_array = np.fromiter(SIC_letter_MS_frac.values(),
                                       dtype='float')

    # --- Convert fractional employment to GVA ---

    # Load GVA table: ONS Table 1c from GVA by industry, region
    GVA = pd.read_csv('../data/raw/GVA/SIC_GVA_matrix.csv')

    SIC07_2d = []
    # Ensure all codes are formatted as 2 digits (e.g. '01' rather than '1')
    for code in GVA['SIC07'].values:
        if len(code) == 1:
            code = '0' + code
        SIC07_2d.append(code)
    GVA['SIC07'] = SIC07_2d

    # Estimate GVA: MS jobs not mapped to any economically-active sectors
    # In practice, mostly associated with industry letter code U
    MS_jobs_unmapped = np.sum(SIC_MS) - MS_jobs_mapped_into_fracs
    if DEBUG:
        print('MS jobs unmapped:', MS_jobs_unmapped)

    # Compute GVA due to MSR via SIC2d_MS_frac
    GVA_from_MS = {code: 0 for code in SIC2d_MS_frac}

    for code in SIC2d_MS_frac:
        if code == '99':  # non-classifiable Establishments
            continue
        frac = SIC2d_MS_frac[code]
        GVA_from_code = GVA[gva_year][GVA['SIC07'] == code].values[0]*1e6
        if not np.isnan(SIC2d_MS_frac[code]):
            GVA_from_MS[code] = GVA_from_code*SIC2d_MS_frac[code]

    GVA_from_MS_mapped = sum(GVA_from_MS.values())

    if VERBOSE > 1:
        print(f'\n{"="*70}\nConfiguration: {k}\n{"="*70}')
        print(f'GVA re-app.\tWorkers\t\tDirect GVA\tProductivity\n{"-"*70}')

    for gva_unmapped in ['None', 'UK avg.', 'MS avg.']:
        if gva_unmapped == 'UK avg.':
            # Re-apportion unmapped workers via UK-wide average productivity
            GVA_from_MS_unmapped = MS_jobs_unmapped/(wf_size['2022']*1e6
                                                     )*np.sum(GVA[gva_year])*1e9
        elif gva_unmapped == 'MS avg.':
            # Re-apportion unmapped workers via MS average productivity
            GVA_from_MS_unmapped = MS_jobs_unmapped*(
                GVA_from_MS_mapped/MS_jobs_mapped_into_fracs)
        else:
            # Assign zero GVA to unmapped jobs
            GVA_from_MS_unmapped = 0

        # GVA computed for intermediate year: 2022 if final year is 2023,
        # or 2011 if final year is 2010, for reasons explained above
        GVA_MS_inter = (GVA_from_MS_mapped + GVA_from_MS_unmapped)

        # --- Convert 2022 GVA to 2023 terms ---

        if OUTPUT_YEAR == '2023':
            # Re-scale 2022 GVA to 2023 GVA via ONS numbers; convert to £ bn
            GVA_MS_final = UK_GVA['2023']/UK_GVA['2022']*GVA_MS_inter/1e9
        elif OUTPUT_YEAR == '2010':
            # Re-scale 2011 GVA to 2010 GVA via ONS numbers; convert to £ bn
            GVA_MS_final = UK_GVA['2010']/UK_GVA['2011']*GVA_MS_inter/1e9

        # Productivity per worker
        prod_MS_final = GVA_MS_final*1e9/np.sum(SIC_MS)

        if VERBOSE > 1:
            print(f'{gva_unmapped}\t\t{SIC_MS.sum()/1e6:.2f} mn',
                  f'\t£{GVA_MS_final:.2f} bn \t£{prod_MS_final:,.0f}')

        GVA_all.append(GVA_MS_final)
        prod_all.append(prod_MS_final)
        MS_WF_all.append(np.sum(SIC_MS)/1e6)
        n_config += 1

if VERBOSE > 0:

    print(f'\n{"="*70}\nAll MS workforce configurations (total: {n_config})\n{"="*70}')
    print(f'Distribution\tMean ± std.\t\tPercentiles [16, 50, 84]th\n{"-"*70}')
    print(f'Workforce:\t{np.mean(MS_WF_all):,.2f} ± {np.std(MS_WF_all):,.2f} mn', end='')
    print('\t\t[{0:,.2f}; {1:,.2f}; {2:,.2f}] mn'.format(*pctl(MS_WF_all, [16, 50, 84])))
    print(f'Direct GVA:\t£{np.mean(GVA_all):,.2f} ± {np.std(GVA_all):.2f} bn', end='')
    print('\t[{0:.2f}; {1:.2f}; {2:.2f}] bn'.format(*pctl(GVA_all, [16, 50, 84])))
    print(f'Productivity:\t£{np.mean(prod_all):,.0f} ± {np.std(prod_all):,.0f}', end='')
    print('\t[{0:,.0f}; {1:,.0f}; {2:,.0f}]'.format(*pctl(prod_all, [16, 50, 84])))
    print(f'{"-"*70}')
    print(f'\n{"="*40}\nUK as a whole (for comparison)\n{"="*40}')
    print(f'People in jobs:\t\t{wf_size[OUTPUT_YEAR]/1000:,.2f} mn')
    print(f'Total GVA:\t\t£{UK_GVA[OUTPUT_YEAR]:,.2f} bn')
    print(f'Productivity:\t\t£{(1e6*UK_GVA[OUTPUT_YEAR]/wf_size[OUTPUT_YEAR]):,.0f}')
    print(f'{"-"*40}')

print('\n...done')

GVA_dict = {'GVA_all': GVA_all, 'prod_all': prod_all, 'MS_WF_all': MS_WF_all}

if SAVE_PICKLE:
    with open(f'../data/processed/GVA_all_{OUTPUT_YEAR}.pickle', 'wb') as handle:
        pickle.dump(GVA_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
