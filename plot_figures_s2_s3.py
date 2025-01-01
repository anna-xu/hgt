import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt

# note: hgt_data.csv is an outputted version of HandgripData.mat from MATLAB
hgt = pd.read_csv('data/hgt_data.csv')

hgt = hgt.fillna(pd.NA)
df = hgt.copy()
df = df.dropna(subset=['Tired', 'Borg'])
df['Condition Order'] = df['ConditionOrder'].map({1: 'negative first', 0: 'positive first'})

# Figures S2 and S3b

def plot_raw_data(yvar, yvar_label, num_bins=10, legend_position='lower left'):
    # Sample data (same as above)
    df['x_bins'] = pd.cut(df['zTrialNum_Session'], bins=num_bins,labels=[str(x) for x in np.arange(1,num_bins+1,1)])
    df_binned = df.groupby(['x_bins', 'FeedbackCondition', 'Condition Order'])[yvar].mean().reset_index()
    df_binned['se'] = df.groupby(['x_bins', 'FeedbackCondition', 'Condition Order'])[yvar].apply(lambda grp: grp.std() / np.sqrt(len(grp.values))).reset_index()[yvar]
    df_binned['std'] = df.groupby(['x_bins', 'FeedbackCondition', 'Condition Order'])[yvar].std().reset_index()[yvar]
    df_binned['count'] = df.groupby(['x_bins', 'FeedbackCondition', 'Condition Order'])[yvar].count().reset_index()[yvar]

    df_binned = df_binned.dropna(subset=[yvar])

    FBs = []
    for x in df_binned['FeedbackCondition'].values:
        if x==1:
            FBs += ['negative']
        elif x==0:
            FBs += ['positive']
    df_binned['Feedback'] = FBs

    COs = []
    for x in df_binned['x_bins'].values:
        if int(x) <= num_bins/2:
            COs += ['negative first']
        elif int(x) > num_bins/2:
            COs += ['positive first']
    df_binned['Condition Order'] = COs

    # plot figure
    plt.figure(figsize=[20,10])
    sns.set_style("whitegrid", {'axes.grid' : False})
    ax=sns.lineplot(data=df_binned, 
                x='x_bins', 
                y=yvar, 
                hue='Feedback',
                style = 'Condition Order',
                markers = 'o',
                markersize = 10,
                linewidth = 3,
                palette=['g', 'r'])

    ax.fill_between(df_binned['x_bins'][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) < 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) < 6)] - df_binned['se'][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) < 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) < 6)] + df_binned['se'][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) < 6)], 
                    color='r', 
                    alpha=0.2)
    ax.fill_between(df_binned['x_bins'][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) >= 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) >= 6)] - df_binned['se'][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) >= 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) >= 6)] + df_binned['se'][(df_binned['Feedback'] == 'negative') & (df_binned['x_bins'].astype(int) >= 6)], 
                    color='r', 
                    alpha=0.2)
    ax.fill_between(df_binned['x_bins'][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) < 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) < 6)] - df_binned['se'][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) < 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) < 6)] + df_binned['se'][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) < 6)], 
                    color='g', 
                    alpha=0.2)
    ax.fill_between(df_binned['x_bins'][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) >= 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) >= 6)] - df_binned['se'][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) >= 6)], 
                    df_binned[yvar][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) >= 6)] + df_binned['se'][(df_binned['Feedback'] == 'positive') & (df_binned['x_bins'].astype(int) >= 6)], 
                    color='g', 
                    alpha=0.2)

    # clean up plot
    sns.despine()
    if legend_position != 'None':
        plt.legend(markerscale = 0.65)
        sns.move_legend(ax, legend_position)
    else:
        plt.legend('', frameon=False)
    plt.xlabel('Trial Bin', fontsize=15, fontweight='roman')
    plt.ylabel(yvar_label, fontsize=15, fontweight='roman')
    plt.xticks(fontsize=12, fontweight='roman')
    plt.yticks(fontsize=12, fontweight='roman')
    plt.savefig(f"{yvar}_time.png")

plot_raw_data('AbsEffort', 'Grip Force', num_bins=10, legend_position='None')
plot_raw_data('Tired', 'Fatigue', num_bins=10, legend_position='lower right')

# Figure S3a

def plot_raw_data_effort_demands(yvar, yvar_label, num_bins=10, legend_position='lower left'):
    df['x_bins'] = pd.cut(df['zCenterBox'], bins=num_bins,labels=[str(x) for x in np.arange(1,num_bins+1,1)])
    df_binned = df.groupby(['x_bins', 'FeedbackCondition'])[yvar].mean().reset_index()
    df_binned['se'] = df.groupby(['x_bins', 'FeedbackCondition'])[yvar].apply(lambda grp: grp.std() / np.sqrt(len(grp.values))).reset_index()[yvar]
    df_binned['std'] = df.groupby(['x_bins', 'FeedbackCondition'])[yvar].std().reset_index()[yvar]
    df_binned['count'] = df.groupby(['x_bins', 'FeedbackCondition'])[yvar].count().reset_index()[yvar]

    FBs = []
    for x in df_binned['FeedbackCondition'].values:
        if x==1:
            FBs += ['negative']
        elif x==0:
            FBs += ['positive']
    df_binned['Feedback'] = FBs

    # plot figure
    plt.figure(figsize=[20,10])
    sns.set_style("whitegrid", {'axes.grid' : False})
    ax=sns.lineplot(data=df_binned, 
                x='x_bins', 
                y=yvar, 
                hue='Feedback',
                markers = 'o',
                markersize = 10,
                linewidth = 3,
                palette=['g', 'r'])

    ax.fill_between(df_binned['x_bins'][df_binned['Feedback'] == 'negative'], 
                    df_binned[yvar][df_binned['Feedback'] == 'negative'] - df_binned['se'][df_binned['Feedback'] == 'negative'], 
                    df_binned[yvar][df_binned['Feedback'] == 'negative'] + df_binned['se'][df_binned['Feedback'] == 'negative'], 
                    color='r', 
                    alpha=0.2)

    ax.fill_between(df_binned['x_bins'][df_binned['Feedback'] == 'positive'], 
                    df_binned[yvar][df_binned['Feedback'] == 'positive'] - df_binned['se'][df_binned['Feedback'] == 'positive'], 
                    df_binned[yvar][df_binned['Feedback'] == 'positive'] + df_binned['se'][df_binned['Feedback'] == 'positive'], 
                    color='g', 
                    alpha=0.2)

    # connect the two lines
    df_connect = df_binned[df_binned['x_bins'].isin([str(int(num_bins/2)),str(int((num_bins/2)+1))])]

    plt.plot(df_connect['x_bins'][df_connect['Feedback']=='positive'], 
            df_connect[yvar][df_connect['Feedback'] == 'positive'],
            marker = 'o',
            markersize=8,
            color = 'green',
            linewidth=3)

    plt.plot(df_connect['x_bins'][df_connect['Feedback']=='negative'], 
            df_connect[yvar][df_connect['Feedback'] == 'negative'],
            marker = 'o',
            markersize=8,
            color = 'red',
            linewidth=3)

    # clean up plot
    sns.despine()
    if legend_position != 'None':
        plt.legend(markerscale = 0.65)
        sns.move_legend(ax, legend_position)
    else:
        plt.legend('', frameon=False)
    plt.xlabel('Effort Demand Bin', fontsize=15, fontweight='roman')
    plt.ylabel(yvar_label, fontsize=15, fontweight='roman')
    plt.xticks(fontsize=12, fontweight='roman')
    plt.yticks(fontsize=12, fontweight='roman')
    plt.savefig(f"{yvar}_effort-demand.png")

plot_raw_data_effort_demands('AbsEffort', 'Grip Force', num_bins=10, legend_position='None')