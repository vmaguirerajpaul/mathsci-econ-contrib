#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute all SOC re-basing mappings, incl. ONS's own mapping from
dual-coding exercises.
"""

__author__ = 'Vinesh Maguire Rajpaul'
__email__ = 'vr325@cantab.ac.uk'
__status__ = 'Development'

import copy
import pickle
import pandas as pd
import numpy as np


def flatten(xss):
    """
    Helper function to convert list of lists (or arrays of arrays) to
    1D list; possible equivalent: np.concatenate(xss).ravel().tolist()
    """
    return [x for xs in xss for x in xs]


def main():

    # Load dictionaries corresponding to ASHE tables [ONS]
    with open('../data/processed/ASHE_dicts_all.pickle', 'rb') as input_file:
        ASHE_dicts_all = pickle.load(input_file)

    # Load dictionary corresponding to Deloitte's published data
    with open('../data/processed/D2010_data.pickle', 'rb') as input_file:
        D2010 = pickle.load(input_file)

    # Load ONS SOC relationship dictionaries
    with open('../data/processed/ONS_SOC_relationships.pickle', 'rb'
              ) as input_file:
        SOC_relations = pickle.load(input_file)

    SOC_INPUTS = ['2000', '2010']
    SOC_OUTPUTS = ['2010', '2020']

    ASHE_IN_TABLES = ['2010r', '2011r']
    ASHE_OUT_TABLES = ['2011r', '2023p']

    # Use most complete ASHE tables for SOC re-mapping: hourly rather
    # than annual & full-time plus part-time instead of full-time only
    HRAN_FTPT = 'hourly_FTPT'

    # Remove SOC2000 codes with 0 MS jobs (per Deloitte) from MS code list
    MS_SOC2000_codes = sorted(list(set(D2010['SOC2000_code']) - set(
        D2010['SOC2000_code'][D2010['MS_jobs'] == 0])))

    # Define arrays containing MS codes in inout SOC scheme
    MS_CODES_INPUT = [MS_SOC2000_codes,
                      # placeholder: 2011 mapped jobs (computed in loop)
                      np.nan
                      ]

    # Define SOC codes that are definitely not MS jobs, despite some MS
    # jobs in SOC2000 mapping to these codes in SOC2010 or SOC2020.

    # manually checked: soc2010indexversion5.116august2016.xlsx
    #NON_MS_CODES_SOC2010 =  set([]);
    NON_MS_CODES_SOC2010 = set(['2214', '2215', '2231', '2232', '2431', '2473',
                                '3131', '3132', '3212', '3231', '3411', '3412', '3416', '3513', '3531',
                                '4123', '4217', '5112', '5223', '5225', '5421', '5422', '5423', '5432',
                                '5441', '7121', '7125', '9219'])

    # manually checked: soc2020volume2thecodingindexexcel22022024.xlsx
    #NON_MS_CODES_SOC2020 = set([])
    NON_MS_CODES_SOC2020 = set(['2231', '2232', '2237', '2252', '2494', '3131',
                                '3132', '3211', '3214', '3221', '3411', '3412', '3416', '3541', '4123',
                                '5112', '5223', '5225', '5243', '5245', '5315', '5421', '5422', '5423',
                                '5432', '5441', '7129', '9219'])

    mappings_all = {}

    print('Computing alternative SOC re-basing schemes...', end='')

    for mapping_ix, (MS_codes_in, SOC_in, SOC_out, ASHE_in_tab, ASHE_out_tab
                     ) in enumerate(zip(MS_CODES_INPUT, SOC_INPUTS,
                                        SOC_OUTPUTS, ASHE_IN_TABLES, ASHE_OUT_TABLES)):

        if SOC_in == '2000':
            map_matrix = pd.read_csv('../data/raw/ONS_SOC2000_SOC2010/SOC2000_SOC2010_v7.csv',
                                     encoding_errors='ignore')
        else:
            map_matrix = pd.read_csv('../data/raw/ONS_SOC2010_SOC2020/SOC2010_SOC2020_v10.csv',
                                     encoding_errors='ignore')
            # for second iter, replace MS_codes_in with one generated in loop
            MS_codes_in = mappings_all['2000_2010']['MS_codes_out']

        # no of input codes
        n = len(MS_codes_in)

        # empty array of lists to store mapping, description
        mapping = np.array([[]]*n + [[1]], dtype=object)[:-1]
        mapping_descr_out = np.array([[]]*n + [[1]], dtype=object)[:-1]

        # - type A wght: no. input codes that map backwards from output
        #   code to an input code, e.g. from SOC2010 to SOC2000 code
        # - type B wght: no. actual jobs (ASHE) in output occupation;
        # - type C wght: SOC relationship tables from ONS; ref -
        #   https://tinyurl.com/32s7kxr5 (NB: default in methodology)
        weightsA = np.array([[]]*n + [[1]], dtype=object)[:-1]
        weightsB, weightsC = copy.deepcopy(weightsA), copy.deepcopy(weightsA)

        # cardinality: no. output codes to which an input code is mapped
        cardinality = np.zeros(n)

        # ASHE dict keys: SOC labels for different occupations
        SOC_in_key = 'SOC' + SOC_in + '_code'
        SOC_out_key = 'SOC' + SOC_out + '_code'

        # dict of type C weights [already computed from ONS tables]
        mdc = SOC_relations[SOC_in + '_' + SOC_out]

        ASHE_jobs_in = ASHE_dicts_all[ASHE_in_tab + '_' + HRAN_FTPT]
        ASHE_jobs_out = ASHE_dicts_all[ASHE_out_tab + '_' + HRAN_FTPT]

        mapping_descr_in = [(i, ASHE_jobs_in['description'][
            ASHE_jobs_in[SOC_in_key] == i][0]) for i in MS_codes_in]

        for ix, code in enumerate(MS_codes_in):

            code_ix = np.where(map_matrix['SOC ' + SOC_in].values == code)[0]

            mapped_codes = map_matrix['SOC ' + SOC_out].values[code_ix]

            mapped_codes_unique = set(mapped_codes)
            mapped_codes_unique.update(
                SOC_relations[f'{SOC_in}_{SOC_out}'][code]['output_codes'])

            if SOC_out == '2010':
                mapped_codes_unique.difference_update(NON_MS_CODES_SOC2010)
            elif SOC_out == '2020':
                mapped_codes_unique.difference_update(NON_MS_CODES_SOC2020)

            # for 2010 to 2020 mapping, ensure we only consider SOC2020
            # codes that are associated with MS profession in SOC2000 [1]
            if mapping_ix == 2:
                mapped_codes_unique = mapped_codes_unique & set(
                    mappings_all['2000_2020']['MS_codes_out'])

            # Sort mapped codes numerically
            mapped_codes_unique = sorted(list(mapped_codes_unique))

            cardinality[ix] = len(mapped_codes_unique)
            mapping[ix] = mapped_codes_unique
            mapping_descr_out[ix] = [(i, ASHE_jobs_out['description'][
                ASHE_jobs_out[SOC_out_key] == i][0]) for i in mapped_codes_unique]

            mapped_weightsA = np.zeros(len(mapped_codes_unique))
            mapped_weightsB = np.zeros(len(mapped_codes_unique))
            mapped_weightsC = np.zeros(len(mapped_codes_unique))

            for mapped_code_ix, mapped_code in enumerate(mapped_codes_unique):
                mapped_weightsA[mapped_code_ix] = np.sum(
                    mapped_codes == mapped_code)
                mapped_weightsB[mapped_code_ix] = ASHE_jobs_out['no_jobs'][
                    ASHE_jobs_out[SOC_out_key] == mapped_code]
                mapped_weightsC[mapped_code_ix] = (mdc[code]['weights'][
                    mdc[code]['output_codes'] == mapped_code].sum())

            mapped_weightsB[np.isnan(mapped_weightsB)] = 1
            mapped_weightsC[np.isnan(mapped_weightsC)] = 1

            weightsA[ix] = mapped_weightsA
            weightsB[ix] = mapped_weightsB
            weightsC[ix] = mapped_weightsC

        # codes that end up in output SOC scheme as MS codes
        flat_list = flatten(mapping)
        MS_codes_out = sorted(np.unique(flat_list))

        # MS output codes & job descriptions (for quick inspection)
        MS_codes_out_descr = [
            (i, ASHE_jobs_out['description'][np.where(
                ASHE_jobs_out[SOC_out_key] == i)[0][0]])
            for i in MS_codes_out]

        # Store entire mapping (with all possible weights) in dictionary
        mapping_dict = ({
            'SOC_in': SOC_in, 'SOC_out': SOC_out,
            'ASHE_in_tab': ASHE_in_tab, 'ASHE_out_tab': ASHE_out_tab,
            'mapping_descr_in': mapping_descr_in, 'mapping': mapping,
            'mapping_descr_out': mapping_descr_out,
            'cardinality': cardinality,
            'weightsA': weightsA, 'weightsB': weightsB, 'weightsC': weightsC,
            'MS_codes_in': MS_codes_in, 'MS_codes_out': MS_codes_out,
            'MS_codes_out_descr': MS_codes_out_descr
        })
        mappings_all[SOC_in + '_' + SOC_out] = mapping_dict

    with open('../data/processed/SOC_SOC_mappings.pickle', 'wb') as handle:
        pickle.dump(mappings_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('done.')



def print_mapping(md):
    # Currently for debugging/inspection purposes only

    for ix, (map_in, map_out) in enumerate(zip(md['mapping_descr_in'],
                                               md['mapping_descr_out']
                                               )):
        print(ix, '---', map_in, '---')
        print()
        print('>>>', map_out)
        print()


def mapping_to_matrix(mapping_dict, wght_type='C'):
    # Convert mapping lists stored in dictionary to mapping matrix

    md = mapping_dict

    # M, N: matrix will have dimension (M, N)
    M, N = len(md['MS_codes_in']), len(md['MS_codes_out'])

    # Check that SOC codes are sorted numerically
    assert(np.all(sorted(md['MS_codes_in']) == md['MS_codes_in']))
    assert(np.all(sorted(md['MS_codes_out']) == md['MS_codes_out']))

    # Row and column labels for the transformation matrix
    col_labels = {v: i for i, v in enumerate(md['MS_codes_in'])}
    row_labels = {v: i for i, v in enumerate(md['MS_codes_out'])}

    # Initialise re-basing matrix S
    S = np.zeros((N, M))

    # Define initial weighting for each non-zero element in S
    for i, col_code in enumerate(col_labels):
        initial_weight = 1  # equal weighting
        for j, mapped_code in enumerate(md['mapping'][i]):
            #print(i,j,mapped_code, len(md['weightsC'][i]))
            if (len(md['weightsA'][i]) > 0) and (wght_type == 'A'):
                initial_weight = md['weightsA'][i][j]
            elif (len(md['weightsB'][i]) > 0) and (wght_type == 'B'):
                initial_weight = md['weightsB'][i][j]
            elif (len(md['weightsC'][i]) > 0) and (wght_type == 'C'):
                initial_weight = md['weightsC'][i][j]
            elif wght_type == 'random':
                initial_weight = np.random.rand()
            S[row_labels[mapped_code], i] = initial_weight

    # Renormalise matrix S: all columns should sum to unity
    for i, col_code in enumerate(col_labels):
        if np.nansum(S[:, i]) == 0:
            print(i, wght_type, col_code)  # debugging
        S[:, i] /= np.nansum(S[:, i])

    # Zero weight for all occupation re-basing pairs with NaN weights
    S[np.isnan(S)] = 0
    return S


if __name__ == "__main__":
    main()

# %% --- Notes to self ---
"""
[1] To compute codes that are mapped from occupations identified as MS
    in SOC2010 to occupations in SOC2020, but are not mapped back to any
    MS occupation in SOC2000 (and therefore not guaranteed to be MS):
        
            ``` 
            setSmall = set(sorted(mappings_all['2000_2020']['codes_MS']))
            setBig = set(sorted(mappings_all['2010_2020']['codes_MS']))
            print(setBig-setSmall)
            ```
            
    --> {'3554', '1135', '3582', '3120', '1251', '5119', '5321', '7131', 
         '3413', '8119', '2324', '3544', '4112', '4217', '7121', '6211', 
         '8135', '3543', '4111', '8160', '1256', '2142', '8149', '3231', 
         '3557', '1257', '7125', '5233', '3553', '3219', '1224', '3555', 
         '6135', '1233', '1255'}
    
    All of these are very clearly NOT MS (e.g. 5321 - plasters), with an
    exception:
        
        3544 Data analysts
        
    So this could be manually permitted in the SOC2010 to SOC2020
    mapping.Note however that the corresponding SOC2000 code, 3539, was
    not deemed MS by Deloitte, so the SOC2000 --> SOC2020 mapping will
    not capture any data analysts.


[2] #set(SOC_SOC_mappings['2000_2020']['MS_codes_out']) - 
    set(SOC_SOC_mappings['2010_2020']['MS_codes_out']) 
    
    # check for consistency of mappings
        
"""
