"""
Author: Shou-Han Zhou
Email: shou-han.zhou@monash.edu
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hddm
from joblib import Parallel, delayed
from IPython import embed as shell
from sklearn.metrics import r2_score
from scipy import stats


plt.close('all')

def simulate_data(a, v, t, z=0.5, sv=0, sz=0, st=0, condition=0, nr_trials1=1000, nr_trials2=1000):
    parameters1 = {'a':a, 'v':v, 't':t, 'z':z, 'sv':sv, 'sz': sz, 'st': st}
    parameters2 = {'a':a, 'v':v, 't':t, 'z':1-z, 'sv':sv, 'sz': sz, 'st': st}
    df_sim1, params_sim1 = hddm.generate.gen_rand_data(params=parameters1, size=nr_trials1, subjs=1, subj_noise=0)
    df_sim1['condition'] = 1
    df_sim2, params_sim2 = hddm.generate.gen_rand_data(params=parameters2, size=nr_trials2, subjs=1, subj_noise=0)
    df_sim2['condition'] = 0
    df_sim = pd.concat((df_sim1, df_sim2))
    df_sim['correct'] = df_sim['response'].astype(int)
    df_sim['response'] = df_sim['response'].astype(int)
    df_sim['group'] = condition#df_sim = df_sim.rename(columns={'condition':'side'})
    return df_sim

def fit_subject(data, quantiles):
    """
    Simulates stim-coded data.
    """

    subj_idx = np.unique(data['subj_idx'])

    m = hddm.HDDM(data, bias=False)
   # m = hddm.HDDMStimCoding(data, stim_col='side', split_param='v', drift_criterion=False, bias=False, p_outlier=0,
    #                        depends_on={'v':'group', 'a':'group', 't':'group', 'z':'group', })
    m.optimize('gsquare', quantiles=quantiles, n_runs=1000)
    res = pd.concat((pd.DataFrame([m.values], index=[subj_idx]), pd.DataFrame([m.bic_info], index=[subj_idx])), axis=1)
    return res

#######################################################################################################################

os.chdir('C:/SHwork/ManaProject/DDM')
os.getcwd()
df_emp = hddm.load_csv('Data/dataBehaviourAllTrials.csv')
df_emp['correct']=df_emp['response']
df_emp['rt'] = df_emp['rt']/1000
#df_emp = df_emp.drop(columns = ['Unnamed: 8'])
n_subjects = len(np.unique(df_emp['subj_idx']))
# fit chi-square:
quantiles = np.arange(0.1,1,0.05)
dataframes =[]
subjectInterest = {44} #put your participant here
for num in subjectInterest:
    data = df_emp[df_emp['subj_idx']==num]
    gs = fit_subject(data,quantiles)
    gs['group']=data["group"].iloc[0]
    gs['subject'] = data["subjName"].iloc[0]
    dataframes.append(gs)

params_fitted = pd.concat(dataframes, ignore_index= True)
print(params_fitted.head())


#save the files
params_fitted.to_csv('avtGood44.csv', index=False)
###########################################################################################################################################
# posterior
# simulate data based on fitted params:
dfs = []
trials_per_level = 1000
for i in range(len(subjectInterest)):
    df0 = simulate_data(a=params_fitted.loc[i,'a'], v=params_fitted.loc[i,'v'],
                        t=params_fitted.loc[i,'t'],
                         condition=0, nr_trials1=trials_per_level,
                        nr_trials2=trials_per_level)
    df = df0
    df['subj_idx'] = i
    dfs.append(df)
df_sim = pd.concat(dfs)
df_sim.loc[df_sim["response"]==0, 'rt'] = np.NaN
df_simT = df_sim
#################################################################################################################
# PLOT THE DATA:
df_group=df_emp
df_sim_group=df_simT
quantiles=[0, 0.1, 0.3, 0.5, 0.7, 0.9, ]
nr_subjects =5# len(np.unique(df_group['subj_idx']))

plt_nr = 1
fig = plt.figure(figsize=(10, nr_subjects * 2))
ks = -1
for s in subjectInterest:
    ks=ks+1
    if plt_nr>4*nr_subjects:
        fig = plt.figure(figsize=(10, nr_subjects * 2))
        plt_nr = 1

    df = df_group.copy().loc[(df_group['subj_idx'] == s), :]
    df_sim = df_sim_group.copy().loc[(df_sim_group['subj_idx'] == ks), :]
    df['rt_acc'] = df['rt'].copy()
    df.loc[df['correct'] == 0, 'rt_acc'] = df.loc[df['correct'] == 0, 'rt_acc'] * -1
    df['rt_resp'] = df['rt'].copy()
    df.loc[df['response'] == 0, 'rt_resp'] = df.loc[df['response'] == 0, 'rt_resp'] * -1
    df_sim['rt_acc'] = df_sim['rt'].copy()
    df_sim.loc[df_sim['correct'] == 0, 'rt_acc'] = df_sim.loc[df_sim['correct'] == 0, 'rt_acc'] * -1
    df_sim['rt_resp'] = df_sim['rt'].copy()
    df_sim.loc[df_sim['response'] == 0, 'rt_resp'] = df_sim.loc[df_sim['response'] == 0, 'rt_resp'] * -1
    max_rt = np.percentile(df_sim.loc[~np.isnan(df_sim['rt']), 'rt'], 99)
    bins = np.linspace(-max_rt, max_rt, 40)

    # rt distributions correct vs error:
    ax = fig.add_subplot(nr_subjects, 4, plt_nr)
    ax.hist(df.loc[:, 'rt_acc'], bins=bins,
                               density=True, color='green', alpha=0.5)
    ax.hist(df_sim.loc[:, 'rt_acc'], bins=bins, density=True,
                histtype='step', color='k', alpha=1, label=None)
    ax.set_title('P(correct)={}'.format(round(df.loc[:, 'correct'].mean(), 3), ))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Trials (prob. dens.)')
    plt_nr += 1

    # condition accuracy plots:
    ax = fig.add_subplot(nr_subjects, 4, plt_nr)
    df.loc[:, 'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    d = df.groupby(['rt_bin']).mean().reset_index()
    ax.errorbar(d.loc[:, "rt"], d.loc[:, "correct"], fmt='-o', color='orange', markersize=10)
    df_sim.loc[:, 'rt_bin'] = pd.qcut(df_sim['rt'], quantiles, labels=False)
    d = df_sim.groupby(['rt_bin']).mean().reset_index()
    ax.errorbar(d.loc[:, "rt"], d.loc[:, "correct"], fmt='x', color='k', markersize=6)
    ax.set_ylim(0, 1)
    ax.set_title('Conditional accuracy')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(correct)')
    plt_nr += 1

    # condition accuracy plots:
    ax = fig.add_subplot(nr_subjects, 4, plt_nr)
    if np.isnan(df['rt']).sum() > 0:
        bar_width = 1
        fraction_yes = df['response'].mean()
        fraction_yes_sim = df_sim['response'].mean()
        hist, edges = np.histogram(df.loc[:, 'rt_resp'], bins=bins, density=True, )
        hist = hist * fraction_yes
        hist_sim, edges_sim = np.histogram(df_sim.loc[:, 'rt_resp'], bins=bins, density=True, )
        hist_sim = hist_sim * fraction_yes_sim
        ax.bar(edges[:-1], hist, width=np.diff(edges)[0], align='edge',
               color='magenta', alpha=0.5, linewidth=0, )
        # ax.plot(edges_sim[:-1], hist_sim, color='k', lw=1)
        ax.step(edges_sim[:-1] + np.diff(edges)[0], hist_sim, color='black', lw=1)
        # ax.hist(hist, edges, histtype='stepfilled', color='magenta', alpha=0.5, label='response')
        # ax.hist(hist_sim, edges_sim, histtype='step', color='k',)
        no_height = (1 - fraction_yes) / bar_width
        no_height_sim = (1 - fraction_yes_sim) / bar_width
        ax.bar(x=-1.5, height=no_height, width=bar_width, alpha=0.5, color='cyan', align='center')
        ax.hlines(y=no_height_sim, xmin=-2, xmax=-1, lw=0.5, colors='black', )
        ax.vlines(x=-2, ymin=0, ymax=no_height_sim, lw=0.5, colors='black')
        ax.vlines(x=-1, ymin=0, ymax=no_height_sim, lw=0.5, colors='black')
    else:
        N, bins, patches = ax.hist(df.loc[:, 'rt_resp'], bins=bins,
                                   density=True, color='magenta', alpha=0.5)
        ax.hist(df_sim.loc[:, 'rt_resp'], bins=bins, density=True,
                histtype='step', color='k', alpha=1, label=None)
    ax.set_title('P(bias)={}'.format(round(df.loc[:, 'response'].mean(), 3), ))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Trials (prob. dens.)')
    plt_nr += 1

    # condition response plots:
    ax = fig.add_subplot(nr_subjects, 4, plt_nr)
    df.loc[:, 'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    d = df.groupby(['rt_bin']).mean().reset_index()
    ax.errorbar(d.loc[:, "rt"], d.loc[:, "response"], fmt='-o', color='orange', markersize=10)
    df_sim.loc[:, 'rt_bin'] = pd.qcut(df_sim['rt'], quantiles, labels=False)
    d = df_sim.groupby(['rt_bin']).mean().reset_index()
    ax.errorbar(d.loc[:, "rt"], d.loc[:, "response"], fmt='x', color='k', markersize=6)
    ax.set_ylim(0, 1.2)
    ax.set_title('Conditional response')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(bias)')
    plt_nr += 1

sns.despine(offset=5, trim=True)
plt.show()
