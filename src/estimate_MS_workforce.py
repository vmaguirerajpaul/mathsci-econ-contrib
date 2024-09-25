#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update MS workforce estimates from 2010 to 2023 via ONS data
"""

__author__ = 'Vinesh Maguire Rajpaul'
__email__ = 'vr325@cantab.ac.uk'
__status__ = 'Development'

import copy
import itertools
import pickle
import pandas as pd
import numpy as np

from compute_all_SOC_maps import mapping_to_matrix,  print_mapping, flatten

print('Updating mathematical science workforce from 2010 to 2023...')

# whether to save output to disk
SAVE_PICKLE = True

# whether to print output to screen
VERBOSE = True

# Possible SOC re-basing weighting schemes to use
WGHT_TYPE_ALL = ['A', 'B', 'C', 'equal']
# Whether to use ASHE records where hourly or annual pay data are
# available (the former cover the entire sample; the latter do not)
HRAN_ALL = ['hourly', 'annual']
# Whether to include part-time (FTPT) or use full-time only (FT) data
FTPT_ALL = ['FTPT', 'FT']

# Load dictionaries corresponding to ASHE tables [ONS]
with open('../data/processed/ASHE_dicts_all.pickle', 'rb') as input_file:
    ASHE_dicts_all = pickle.load(input_file)

# Load dictionary corresponding to Deloitte's published data
with open('../data/processed/D2010_data.pickle', 'rb') as input_file:
    D2010 = pickle.load(input_file)

# Load SOC-SOC transforms
with open('../data/processed/SOC_SOC_mappings.pickle', 'rb') as input_file:
    SOC_SOC_mappings = pickle.load(input_file)

# Total numbers of UK workers in various years
# Ref: https://tinyurl.com/y62x58nf [ONS]
wf_size = pd.read_csv('../data/raw/people_in_jobs.csv')
wf_size = {str(y): j for (y, j) in zip(wf_size['year'].values,
                                       wf_size['jobs'].values)}



def rescale_within_SOC(
        MSW_DF_in, ASHE_in, ASHE_out_p, ASHE_out_r, SOC_in_out, nan_scaling):

    """
    Rescale workforce across years when SOC framework is unchanged;
    i.e. use changes in individual occupation numbers from ASHE to
    re-scale workforce on an occupation-by-occupation basis.
    Suffix p/r: provisional, revised.
    """

    MSW_DF_out = copy.deepcopy(MSW_DF_in)

    # Tracks how many NaNs encountered during re-scaling (debugging)
    nancount = 0

    for ix, record in enumerate(MSW_DF_in.values):
        job_code = record[0]

        # Work out the top and bottom line of the division, noting that
        # these are one item vectors, so pick the [0] item specifically
        # to avoid a deprecation when converting to np.double
        numerator = ASHE_out_p['no_jobs'][ASHE_out_p[SOC_in_out + '_code'] == job_code][0]
        denominator = ASHE_in['no_jobs'][ASHE_in[SOC_in_out + '_code'] == job_code][0]
        scale_factor_job = np.double(numerator / denominator)

        # If job info not available in input and/or output years,
        # re-scale by overall workforce scaling between the two years
        if np.isnan(scale_factor_job):
            nancount += 1
            scale_factor_job = copy.deepcopy(nan_scaling)

        # TODO: This line causes a deprecation warning
        # Setting an item of incompatible dtype is deprecated and will
        # raise an error in a future version of pandas.
        # e.g. Value '[402.08019707]' has dtype incompatible with int64,
        # please explicitly cast to a compatible dtype first.
        MSW_DF_out.loc[MSW_DF_out['SOC'] == job_code, 'MS_jobs'] *= (
            scale_factor_job)

    # Correct provisional to revised totals
    MSW_DF_out['MS_jobs'] *= ASHE_out_r['no_jobs'][0]/ASHE_out_p['no_jobs'][0]

    return MSW_DF_out, nancount


# Dict: store all mathematical science job (MSW) DataFrame dicts
all_MSW_dicts = {}

# Iterate over all combinations of weights, hourly vs annual data,
# full-time vs part-time data, via Cartesian product
# Ref: https://tinyurl.com/38uwrkhc
all_configs = [WGHT_TYPE_ALL, HRAN_ALL, FTPT_ALL]

for ix, element in enumerate(itertools.product(*all_configs)):

    wght_type, hran_label, ftpt_label = element
    config_label = '_'.join([wght_type, hran_label, ftpt_label])

    all_years = ['2010r', '2011p', '2011r', '2021p', '2021r', '2023p']
    all_dicts = [i + '_' + hran_label + '_' + ftpt_label for i in all_years]
    S2010r, S2011p, S2011r, S2021p, S2021r, S2023p = (
        ASHE_dicts_all[ad] for ad in all_dicts)

    # --- Scale MS jobs from 2010 to 2011 [SOC2000] ---

    MSW_2010_SOC2000 = pd.DataFrame(
        {'SOC': D2010['SOC2000_code'][D2010['MS_jobs'] > 0],
         'MS_jobs': D2010['MS_jobs'][D2010['MS_jobs'] > 0]})

    MSW_2011_SOC2000, _ = rescale_within_SOC(MSW_2010_SOC2000,
                                             S2010r, S2011p, S2011r, 'SOC2000',
                                             wf_size['2011']/wf_size['2010'])

    # --- Re-base 2011 MS numbers from S2000 to S2010 numbers ---

    md_use = SOC_SOC_mappings['2000_2010']
    SOC_SOC_mappings_used = {'2000_2010': md_use}

    S_SOC2000_to_SOC2010 = mapping_to_matrix(md_use, wght_type=wght_type)
    MSW_2011_SOC2010 = S_SOC2000_to_SOC2010@MSW_2011_SOC2000['MS_jobs'].values

    MSW_2011_SOC2010 = pd.DataFrame({
        'SOC': [i for i in md_use['MS_codes_out']],
        'Description': [k[1] for k in md_use['MS_codes_out_descr']],
        'MS_jobs': MSW_2011_SOC2010})

    # --- Scale MS jobs from 2011 to 2021 [S2010] ---

    MSW_2021_SOC2010, _ = rescale_within_SOC(MSW_2011_SOC2010, S2011r,
                                             S2021p, S2021r, 'SOC2010',
                                             wf_size['2021']/wf_size['2011'])

    MSW_2021_SOC2010['2021_2011_growth_prop'] = (
        MSW_2021_SOC2010['MS_jobs']/(MSW_2011_SOC2010['MS_jobs']))
    MSW_2021_SOC2010['2021_2011_growth_abs'] = (
        MSW_2021_SOC2010['MS_jobs']-(MSW_2011_SOC2010['MS_jobs']))

    # --- Re-base 2021 MS numbers from S2010 to S2020 ---

    md_use = SOC_SOC_mappings['2010_2020']
    SOC_SOC_mappings_used['2010_2020'] = md_use

    S_SOC2010_to_SOC2020 = mapping_to_matrix(md_use,
                                             wght_type=wght_type)
    MSW_2021_SOC2020 = S_SOC2010_to_SOC2020@MSW_2021_SOC2010['MS_jobs'].values

    MSW_2021_SOC2020 = pd.DataFrame({
        'SOC': [i for i in md_use['MS_codes_out']],
        'Description': [k[1] for k in md_use['MS_codes_out_descr']],
        'MS_jobs': MSW_2021_SOC2020})

    # --- Scale MS jobs from 2021 to 2023 [S2020] ---

    MSW_2023_SOC2020, _ = rescale_within_SOC(MSW_2021_SOC2020, S2021r,
                                             S2023p, S2023p, 'SOC2020',
                                             wf_size['2023']/wf_size['2021'])

    MSW_2023_SOC2020['2023_2021_growth_prop'] = (
        MSW_2023_SOC2020['MS_jobs']/(MSW_2021_SOC2020['MS_jobs']))
    MSW_2023_SOC2020['2023_2021_growth_abs'] = (
        MSW_2023_SOC2020['MS_jobs']-(MSW_2021_SOC2020['MS_jobs']))

    # --- Re-base 2023 MS numbers from S2020 to S2010 (for SIC/GVA calc) ---

    S_SOC2020_to_SOC2010 = np.linalg.pinv(S_SOC2010_to_SOC2020)
    MSW_2023_SOC2010 = S_SOC2020_to_SOC2010@MSW_2023_SOC2020['MS_jobs'].values

    MSW_2023_SOC2010 = pd.DataFrame({
        'SOC': SOC_SOC_mappings_used['2010_2020']['MS_codes_in'],
        'Description': [i[1] for i in SOC_SOC_mappings_used['2010_2020'][
            'mapping_descr_in']],
        'MS_jobs': MSW_2023_SOC2010})

    # Renormalise to ensure same totals in S2020 and S2010
    renorm_factor = (sum(MSW_2023_SOC2010['MS_jobs'].values) /
                     sum(MSW_2023_SOC2020['MS_jobs'].values))
    MSW_2023_SOC2010['MS_jobs'] *= renorm_factor

    # Naive growth calculation (UK-wide growth only, not SOC-by-SOC)
    UK_scale_factor_2023_2010 = wf_size['2023']/wf_size['2010']
    MSW_2023_naive = (MSW_2010_SOC2000[
        'MS_jobs'].values*UK_scale_factor_2023_2010)

    if VERBOSE:
        print(f'\n{"="*40}\nConfiguration: %s_%s_%s\n{"="*40}'
              % (element))
        print(f'Year\tSOC framework\tWorkforce\n{"-"*40}')
        print(f'2010\tSOC-2000\t{MSW_2010_SOC2000["MS_jobs"].sum():,.0f}')
        print(f'2011\tSOC-2000\t{MSW_2011_SOC2000["MS_jobs"].sum():,.0f}')
        print(f'2011\tSOC-2010\t{MSW_2011_SOC2010["MS_jobs"].sum():,.0f}')
        print(f'2021\tSOC-2010\t{MSW_2021_SOC2010["MS_jobs"].sum():,.0f}')
        print(f'2021\tSOC-2020\t{MSW_2021_SOC2020["MS_jobs"].sum():,.0f}')
        print(f'2023\tSOC-2020\t{MSW_2023_SOC2020["MS_jobs"].sum():,.0f}')
    #print('\t2023 [SOC2020], naive growth):',int(np.nansum(MSW_2023_naive)))

    # Save MS job estimates

    MSW_dfs = ['MSW_2010_SOC2000', 'MSW_2011_SOC2000', 'MSW_2011_SOC2010',
               'MSW_2021_SOC2010', 'MSW_2021_SOC2020', 'MSW_2023_SOC2020',
               'MSW_2023_SOC2010', 'MSW_2023_naive', 'SOC_SOC_mappings_used']
    MSW_dict = {k: eval(k, globals(), locals()) for k in MSW_dfs}

    all_MSW_dicts[config_label] = MSW_dict

print('\n...done.')

if SAVE_PICKLE:
    with open('../data/processed/MSW_all.pickle', 'wb') as handle:
        pickle.dump(all_MSW_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
