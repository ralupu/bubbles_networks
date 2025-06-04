import pandas as pd
import numpy as np
import pysdtest
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

bubble_res = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')
bubble_res = bubble_res[bubble_res['Duration'] > 20]
bubble_res = bubble_res[bubble_res['Duration'] < 100]

b_vc = bubble_res['Firm'].value_counts()
companies_only_one_bubble = list(b_vc[b_vc==1].index.values)
bubble_res = bubble_res[~bubble_res['Firm'].isin(companies_only_one_bubble)]


DF_bubble = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
# DF_bubble.index = pd.to_datetime(DF_bubble['Date'], format = '%d/%m/%Y')

DF_boom = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BMPH (CVM= WB, CVQ=95%, L=0)')
# DF_boom.index = pd.to_datetime(DF_boom['Date'], format = '%d/%m/%Y')

DF_burst = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BRPH (CVM= WB, CVQ=95%, L=0)')
# DF_burst.index = pd.to_datetime(DF_burst['Date'], format = '%d/%m/%Y')

DF_dcovar = pd.read_excel('data_ro/ResultResults_ro_bet_covars.xlsx', sheet_name='Delta CoVaR (K=95%)')


# DF_dcovar.index = pd.to_datetime(DF_dcovar['Date'], format = '%d/%m/%Y')


# get period without bubbles for comparison
def get_period_wout_bubbles(bubbles_company, bubble_length, df_dcovar):
    lead = bubbles_company.shift(-1).dropna()['Start']
    lag = bubbles_company.shift(1).dropna()['End']
    df_diffs = pd.DataFrame(index=bubbles_company.index[1:], columns=['Lead', 'Lag'])
    df_diffs['Lead'] = lead.values
    df_diffs['Lag'] = lag.values
    df_diffs['Diffs'] = lead.values - lag.values
    which_interval_is_max = df_diffs['Diffs'].idxmax()
    end_interval = int(df_diffs['Lead'][which_interval_is_max])
    start_interval = int(df_diffs['Lag'][which_interval_is_max])
    if bubble_length <= end_interval - start_interval + 1:
        edges = end_interval - start_interval - bubble_length - 1
        edge = int(edges / 2)
        start = int(start_interval + edge)
        # end = int(end_interval - edge)
        df_dcovar_comparison = df_dcovar[start:start + bubble_length]
    else:
        df_dcovar_comparison = df_dcovar[start_interval:end_interval]

    return df_dcovar_comparison


# Run stochastic dominance tests
def get_SD_results(bubbles, no_bubbles, company, bubble):
    results_S1 = pd.DataFrame(columns=['BD: Bubble S.D. No-Bubble', 'BD: No-Bubble S.D. Bubble',
                                       'LMW: Bubble S.D. No-Bubble', 'LMW: No-Bubble S.D. Bubble',
                                       'BD paired: Bubble S.D. No-Bubble', 'BD paired: No-Bubble S.D. Bubble',
                                       'LSW: Bubble S.D. No-Bubble', 'LSW: No-Bubble S.D. Bubble',
                                       'DH: Bubble S.D. No-Bubble', 'DH: No-Bubble S.D. Bubble'])
    results_S2 = pd.DataFrame(columns=['BD: Bubble S.D. No-Bubble', 'BD: No-Bubble S.D. Bubble',
                                       'LMW: Bubble S.D. No-Bubble', 'LMW: No-Bubble S.D. Bubble',
                                       'BD paired: Bubble S.D. No-Bubble', 'BD paired: No-Bubble S.D. Bubble',
                                       'LSW: Bubble S.D. No-Bubble', 'LSW: No-Bubble S.D. Bubble',
                                       'DH: Bubble S.D. No-Bubble', 'DH: No-Bubble S.D. Bubble'])
    results_S3 = pd.DataFrame(columns=['BD: Bubble S.D. No-Bubble', 'BD: No-Bubble S.D. Bubble',
                                       'LMW: Bubble S.D. No-Bubble', 'LMW: No-Bubble S.D. Bubble',
                                       'BD paired: Bubble S.D. No-Bubble', 'BD paired: No-Bubble S.D. Bubble',
                                       'LSW: Bubble S.D. No-Bubble', 'LSW: No-Bubble S.D. Bubble',
                                       'DH: Bubble S.D. No-Bubble', 'DH: No-Bubble S.D. Bubble'])

    BD_12_S1 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=1, resampling='bootstrap')
    BD_12_S2 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=2, resampling='bootstrap')
    BD_12_S3 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=3, resampling='bootstrap')
    BD_21_S1 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=1, resampling='bootstrap')
    BD_21_S2 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=2, resampling='bootstrap')
    BD_21_S3 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=3, resampling='bootstrap')
    BD_12_S1.testing()
    BD_12_S2.testing()
    BD_12_S3.testing()
    BD_21_S1.testing()
    BD_21_S2.testing()
    BD_21_S3.testing()

    LMW_12_S1 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=1, resampling='subsampling', b1=20, b2=20)
    LMW_12_S2 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=2, resampling='subsampling', b1=20, b2=20)
    LMW_12_S3 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=3, resampling='subsampling', b1=20, b2=20)
    LMW_21_S1 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=1, resampling='subsampling', b1=20, b2=20)
    LMW_21_S2 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=2, resampling='subsampling', b1=20, b2=20)
    LMW_21_S3 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=3, resampling='subsampling', b1=20, b2=20)
    LMW_12_S1.testing()
    LMW_12_S2.testing()
    LMW_12_S3.testing()
    LMW_21_S1.testing()
    LMW_21_S2.testing()
    LMW_21_S3.testing()

    BDpaired_12_S1 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=1, resampling='paired_bootstrap', nboot=100)
    BDpaired_12_S2 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=2, resampling='paired_bootstrap', nboot=100)
    BDpaired_12_S3 = pysdtest.test_sd(bubbles, no_bubbles, ngrid=100, s=3, resampling='paired_bootstrap', nboot=100)
    BDpaired_21_S1 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=1, resampling='paired_bootstrap', nboot=100)
    BDpaired_21_S2 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=2, resampling='paired_bootstrap', nboot=100)
    BDpaired_21_S3 = pysdtest.test_sd(no_bubbles, bubbles, ngrid=100, s=3, resampling='paired_bootstrap', nboot=100)
    BDpaired_12_S1.testing()
    BDpaired_12_S2.testing()
    BDpaired_12_S3.testing()
    BDpaired_21_S1.testing()
    BDpaired_21_S2.testing()
    BDpaired_21_S3.testing()

    LSW_12_S1 = pysdtest.test_sd_contact(bubbles, no_bubbles, ngrid=100, s=1, resampling='bootstrap')
    LSW_12_S2 = pysdtest.test_sd_contact(bubbles, no_bubbles, ngrid=100, s=2, resampling='bootstrap')
    LSW_12_S3 = pysdtest.test_sd_contact(bubbles, no_bubbles, ngrid=100, s=3, resampling='bootstrap')
    LSW_21_S1 = pysdtest.test_sd_contact(no_bubbles, bubbles, ngrid=100, s=1, resampling='bootstrap')
    LSW_21_S2 = pysdtest.test_sd_contact(no_bubbles, bubbles, ngrid=100, s=2, resampling='bootstrap')
    LSW_21_S3 = pysdtest.test_sd_contact(no_bubbles, bubbles, ngrid=100, s=3, resampling='bootstrap')
    LSW_12_S1.testing()
    LSW_12_S2.testing()
    LSW_12_S3.testing()
    LSW_21_S1.testing()
    LSW_21_S2.testing()
    LSW_21_S3.testing()

    DH_12_S1 = pysdtest.test_sd_SR(bubbles, no_bubbles, ngrid=100, s=1, resampling='bootstrap')
    DH_12_S2 = pysdtest.test_sd_SR(bubbles, no_bubbles, ngrid=100, s=2, resampling='bootstrap')
    DH_12_S3 = pysdtest.test_sd_SR(bubbles, no_bubbles, ngrid=100, s=3, resampling='bootstrap')
    DH_21_S1 = pysdtest.test_sd_SR(no_bubbles, bubbles, ngrid=100, s=1, resampling='bootstrap')
    DH_21_S2 = pysdtest.test_sd_SR(no_bubbles, bubbles, ngrid=100, s=2, resampling='bootstrap')
    DH_21_S3 = pysdtest.test_sd_SR(no_bubbles, bubbles, ngrid=100, s=3, resampling='bootstrap')
    DH_12_S1.testing()
    DH_12_S2.testing()
    DH_12_S3.testing()
    DH_21_S1.testing()
    DH_21_S2.testing()
    DH_21_S3.testing()

    index_name = company + '_' + str(bubble)
    results_S1.loc[index_name, 'BD: Bubble S.D. No-Bubble'] = BD_12_S1.result['pval']
    results_S1.loc[index_name, 'BD: No-Bubble S.D. Bubble'] = BD_21_S1.result['pval']
    results_S1.loc[index_name, 'LMW: Bubble S.D. No-Bubble'] = LMW_12_S1.result['pval']
    results_S1.loc[index_name, 'LMW: No-Bubble S.D. Bubble'] = LMW_21_S1.result['pval']
    results_S1.loc[index_name, 'BD paired: Bubble S.D. No-Bubble'] = BDpaired_12_S1.result['pval']
    results_S1.loc[index_name, 'BD paired: No-Bubble S.D. Bubble'] = BDpaired_21_S1.result['pval']
    results_S1.loc[index_name, 'LSW: Bubble S.D. No-Bubble'] = LSW_12_S1.result['pval']
    results_S1.loc[index_name, 'LSW: No-Bubble S.D. Bubble'] = LSW_21_S1.result['pval']
    results_S1.loc[index_name, 'DH: Bubble S.D. No-Bubble'] = DH_12_S1.result['pval']
    results_S1.loc[index_name, 'DH: No-Bubble S.D. Bubble'] = DH_21_S1.result['pval']

    results_S2.loc[index_name, 'BD: Bubble S.D. No-Bubble'] = BD_12_S2.result['pval']
    results_S2.loc[index_name, 'BD: No-Bubble S.D. Bubble'] = BD_21_S2.result['pval']
    results_S2.loc[index_name, 'LMW: Bubble S.D. No-Bubble'] = LMW_12_S2.result['pval']
    results_S2.loc[index_name, 'LMW: No-Bubble S.D. Bubble'] = LMW_21_S2.result['pval']
    results_S2.loc[index_name, 'BD paired: Bubble S.D. No-Bubble'] = BDpaired_12_S2.result['pval']
    results_S2.loc[index_name, 'BD paired: No-Bubble S.D. Bubble'] = BDpaired_21_S2.result['pval']
    results_S2.loc[index_name, 'LSW: Bubble S.D. No-Bubble'] = LSW_12_S2.result['pval']
    results_S2.loc[index_name, 'LSW: No-Bubble S.D. Bubble'] = LSW_21_S2.result['pval']
    results_S2.loc[index_name, 'DH: Bubble S.D. No-Bubble'] = DH_12_S2.result['pval']
    results_S2.loc[index_name, 'DH: No-Bubble S.D. Bubble'] = DH_21_S2.result['pval']

    results_S3.loc[index_name, 'BD: Bubble S.D. No-Bubble'] = BD_12_S3.result['pval']
    results_S3.loc[index_name, 'BD: No-Bubble S.D. Bubble'] = BD_21_S3.result['pval']
    results_S3.loc[index_name, 'LMW: Bubble S.D. No-Bubble'] = LMW_12_S3.result['pval']
    results_S3.loc[index_name, 'LMW: No-Bubble S.D. Bubble'] = LMW_21_S3.result['pval']
    results_S3.loc[index_name, 'BD paired: Bubble S.D. No-Bubble'] = BDpaired_12_S3.result['pval']
    results_S3.loc[index_name, 'BD paired: No-Bubble S.D. Bubble'] = BDpaired_21_S3.result['pval']
    results_S3.loc[index_name, 'LSW: Bubble S.D. No-Bubble'] = LSW_12_S3.result['pval']
    results_S3.loc[index_name, 'LSW: No-Bubble S.D. Bubble'] = LSW_21_S3.result['pval']
    results_S3.loc[index_name, 'DH: Bubble S.D. No-Bubble'] = DH_12_S3.result['pval']
    results_S3.loc[index_name, 'DH: No-Bubble S.D. Bubble'] = DH_21_S3.result['pval']

    return results_S1, results_S2, results_S3


# Get periods of bubbles and corresponding previous periods without bubbles

companies = DF_bubble.columns[1:]

results_S1 = pd.DataFrame(columns=['BD: Bubble S.D. No-Bubble', 'BD: No-Bubble S.D. Bubble',
                                   'LMW: Bubble S.D. No-Bubble', 'LMW: No-Bubble S.D. Bubble',
                                   'BD paired: Bubble S.D. No-Bubble', 'BD paired: No-Bubble S.D. Bubble',
                                   'LSW: Bubble S.D. No-Bubble', 'LSW: No-Bubble S.D. Bubble',
                                   'DH: Bubble S.D. No-Bubble', 'DH: No-Bubble S.D. Bubble'])
results_S2 = pd.DataFrame(columns=['BD: Bubble S.D. No-Bubble', 'BD: No-Bubble S.D. Bubble',
                                   'LMW: Bubble S.D. No-Bubble', 'LMW: No-Bubble S.D. Bubble',
                                   'BD paired: Bubble S.D. No-Bubble', 'BD paired: No-Bubble S.D. Bubble',
                                   'LSW: Bubble S.D. No-Bubble', 'LSW: No-Bubble S.D. Bubble',
                                   'DH: Bubble S.D. No-Bubble', 'DH: No-Bubble S.D. Bubble'])
results_S3 = pd.DataFrame(columns=['BD: Bubble S.D. No-Bubble', 'BD: No-Bubble S.D. Bubble',
                                   'LMW: Bubble S.D. No-Bubble', 'LMW: No-Bubble S.D. Bubble',
                                   'BD paired: Bubble S.D. No-Bubble', 'BD paired: No-Bubble S.D. Bubble',
                                   'LSW: Bubble S.D. No-Bubble', 'LSW: No-Bubble S.D. Bubble',
                                   'DH: Bubble S.D. No-Bubble', 'DH: No-Bubble S.D. Bubble'])

for company in companies:
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(company)
    df_dcovar = DF_dcovar[company]
    bubbles_company = bubble_res[bubble_res['Firm'] == company]

    for bubble in range(len(bubbles_company)):
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(bubble)
        start = bubbles_company['Start'].iloc[bubble] - 1
        end = bubbles_company['End'].iloc[bubble] - 1
        duration = bubbles_company['Duration'].iloc[bubble]
        dcovar_bubble = df_dcovar.iloc[start:end + 1]
        df_dcovar_comparison = get_period_wout_bubbles(bubbles_company, duration, df_dcovar)

        bubbles = dcovar_bubble.values
        no_bubbles = df_dcovar_comparison.values

        results_S1_company_bubble, results_S2_company_bubble, results_S3_company_bubble = get_SD_results(bubbles,
                                                                                                         no_bubbles,
                                                                                                         company,
                                                                                                         bubble + 1)
        results_S1 = pd.concat([results_S1, results_S1_company_bubble], axis=0)
        results_S2 = pd.concat([results_S2, results_S2_company_bubble], axis=0)
        results_S3 = pd.concat([results_S2, results_S3_company_bubble], axis=0)

results_S1.to_excel('results/results_S1.xlsx')
results_S2.to_excel('results/results_S2.xlsx')
results_S3.to_excel('results/results_S3.xlsx')
