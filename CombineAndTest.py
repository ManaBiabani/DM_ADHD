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
from scipy import stats

import glob

plt.close('all')
def simulate_data(a, v, t, z=0.5, sv=0, sz=0, st=0, condition=0, nr_trials1=1000, nr_trials2=1000):

    """
    Simulates stim-coded data.
    """

    parameters1 = {'a':a, 'v':v, 't':t, 'z':z, 'sv':sv, 'sz': sz, 'st': st}
    parameters2 = {'a':a, 'v':v, 't':t, 'z':1-z, 'sv':sv, 'sz': sz, 'st': st}
    df_sim1, params_sim1 = hddm.generate.gen_rand_data(params=parameters1, size=nr_trials1, subjs=1, subj_noise=0)
    df_sim1['condition'] = 1
    df_sim2, params_sim2 = hddm.generate.gen_rand_data(params=parameters2, size=nr_trials2, subjs=1, subj_noise=0)
    df_sim2['condition'] = 0
    df_sim = pd.concat((df_sim1, df_sim2))
    df_sim['correct'] = df_sim['response'].astype(int)
    df_sim['response'] = df_sim['response'].astype(int)
    df_sim['group'] = condition
    df_sim = df_sim.rename(columns={'condition':'side'})
    return df_sim


csv_file_path = 'C:/SHwork/ManaProject/DDM/GoodFilesMB/*.csv'
csv_files = glob.glob(csv_file_path)

dataFrames = []

for file in csv_files:
    df = pd.read_csv(file)
    dataFrames.append(df)

params_fitted = pd.concat(dataFrames, ignore_index=True)

print(params_fitted)

params_fitted.to_csv('finalParameters.csv')
#a test
group2a = params_fitted[params_fitted['group']==2]["a"]
group1a = params_fitted[params_fitted['group']==1]["a"]
t_stata, p_valuea = stats.ttest_ind(group1a, group2a)

#v test
group2v = params_fitted[params_fitted['group']==2]["v"]
group1v = params_fitted[params_fitted['group']==1]["v"]
t_statv, p_valuev = stats.ttest_ind(group1v, group2v)

#t test
group2t = params_fitted[params_fitted['group']==2]["t"]
group1t = params_fitted[params_fitted['group']==1]["t"]
t_statt, p_valuet = stats.ttest_ind(group1t, group2t)

###########################################################################################################################################
# Comparison Graph
os.chdir('C:/SHwork/ManaProject/DDM')
os.getcwd()
df_emp = hddm.load_csv('Data/dataBehaviourAllTrials.csv')
df_emp['correct']=df_emp['response']
df_emp['rt'] = df_emp['rt']/1000
#df_emp = df_emp.drop(columns = ['Unnamed: 8'])
n_subjects = len(np.unique(df_emp['subj_idx']))

subjectUnique, indices = np.unique(params_fitted['subject'], return_index=True)

subjectUnique = subjectUnique[np.argsort(indices)]
# simulate data based on fitted params:
dfs = []
trials_per_level = 1000
for i in range(n_subjects):
    df0 = simulate_data(a=params_fitted.loc[i,'a'], v=params_fitted.loc[i,'v'],
                        t=params_fitted.loc[i,'t'],
                         condition=0, nr_trials1=trials_per_level,
                        nr_trials2=trials_per_level)
    df = df0
    df['subj_idx'] =subjectUnique[i]
    dfs.append(df)
df_sim = pd.concat(dfs)
df_sim.loc[df_sim["response"]==0, 'rt'] = np.NaN
df_simT = df_sim
df_sim.to_csv('simulatedData.csv')
#################################################################################################################
#
ks= -1
ks_stats=[]
p_values= []
ks_statschis=[]
p_valuechis=[]
subjects = []
for s in np.unique(df_emp['subjName']):
    ks+=1
    dfFit = df_emp.copy().loc[(df_emp['subjName'] == s), :]
    df_simFit = df_simT.copy().loc[(df_simT['subj_idx'] == s), :]
    dfFit['rt_resp'] = dfFit['rt'].copy()
    dfFit.loc[dfFit['response'] == 0, 'rt_resp'] = dfFit.loc[dfFit['response'] == 0, 'rt_resp'] * -1
    df_simFit['rt_resp'] = df_simFit['rt'].copy()
    df_simFit.loc[df_simFit['response'] == 0, 'rt_resp'] = df_simFit.loc[df_simFit['response'] == 0, 'rt_resp'] * -1
    ks_stat, p_value = stats.ks_2samp(dfFit['rt_resp'], df_simFit['rt_resp'])

    max_rt = np.percentile(df_simFit.loc[~np.isnan(df_simFit['rt']), 'rt'], 99)
    min_rt = np.percentile(df_simFit.loc[~np.isnan(df_simFit['rt']), 'rt'], 1)
    bins = np.linspace(min_rt, max_rt, 21)
    obsFreq, obsFreqEdge = np.histogram(dfFit.loc[:, 'rt_resp'], bins=bins, density=True, )
    obsFreqSim, obsFreqSimEdge= np.histogram(df_simFit.loc[:, 'rt_resp'], bins=bins, density=True, )
    ks_statschi, p_valuechi = stats.chisquare(obsFreq, obsFreqSim)

    subjects.append(s)
    ks_stats.append(ks_stat)
    p_values.append(p_value)
    ks_statschis.append(ks_statschi)
    p_valuechis.append(p_valuechi)

results = pd.DataFrame({
    'Subject': subjects,
    'KS_statsChi': ks_statschis,
    'p_valueKsChi': p_valuechis,
})

results.to_csv('FitChiSquared.csv')
#################################################################################################################
# PLOT THE DATA  & FIND THE FIT OF THE FILES:
df_group=df_emp
df_sim_group=df_simT
quantiles=[0, 0.1, 0.3, 0.5, 0.7, 0.9, ]

nr_subjects =4# len(np.unique(df_group['subj_idx']))

plt_nr = 1
fig = plt.figure(figsize=(10, nr_subjects * 2))
ks = 0

for s in np.unique(df_group['subjName']):
    ks+=1

    df = df_group.copy().loc[(df_group['subjName'] == s), :]
    df_sim = df_sim_group.copy().loc[(df_sim_group['subj_idx'] == s), :]
    df['rt_acc'] = df['rt'].copy()
    df.loc[df['correct'] == 0, 'rt_acc'] = df.loc[df['correct'] == 0, 'rt_acc'] * -1
    df['rt_resp'] = df['rt'].copy()
    df.loc[df['response'] == 0, 'rt_resp'] = df.loc[df['response'] == 0, 'rt_resp'] * -1
    df_sim['rt_acc'] = df_sim['rt'].copy()
    df_sim.loc[df_sim['correct'] == 0, 'rt_acc'] = df_sim.loc[df_sim['correct'] == 0, 'rt_acc'] * -1
    df_sim['rt_resp'] = df_sim['rt'].copy()
    df_sim.loc[df_sim['response'] == 0, 'rt_resp'] = df_sim.loc[df_sim['response'] == 0, 'rt_resp'] * -1
    max_rt = np.percentile(df_sim.loc[~np.isnan(df_sim['rt']), 'rt'], 99)
    bins = np.linspace(-max_rt, max_rt, 21)

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
    #ax.set_title('P(bias)={}'.format(round(df.loc[:, 'response'].mean(), 3), ),fontsize=10)
    ax.set_title(f'Subject_{s}', fontsize=10)
    ax.set_xlabel('RT (s)', fontsize=10)
    ax.set_ylabel('Trials (prob. dens.)', fontsize= 10)
    plt_nr += 1

    if (plt_nr>4*nr_subjects) | (ks==79):
        plt.subplots_adjust(hspace=1, wspace=0.4)
        plt.savefig(f'comparisonPlot_{ks+1}.png', format='png')
        fig = plt.figure(figsize=(10, nr_subjects * 2))
        plt_nr = 1
sns.despine(offset=5, trim=True)
plt.show()
