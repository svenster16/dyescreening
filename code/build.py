from cleanupdata import cleanupdata
import experiments
import compoundRetrival
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
from os import path
import numpy as np


def run_experiments():
    # must type in name of each function you want to run
    experiment_functions = ['linear_reg_all_data_all_features', 'linear_reg_all_data_basic_features',
                            'linear_reg_all_data_essential_features', 'linear_reg_all_data_baseline_negative',
                            'linear_reg_dyomics_data_basic_features', 'linear_reg_dyomics_data_all_features',
                            'logit_reg_all_data_all_features', 'logit_reg_all_data_basic_features',
                            'logit_reg_all_data_basic_features_balanced']
    experiment_results = pd.DataFrame()
    for experiment in experiment_functions:
        experiment_to_run = getattr(experiments, experiment)
        experiment_result = experiment_to_run()
        experiment_results = experiment_results.append(experiment_result)
    experiment_results.to_parquet('../results/experiment_results.parquet.gzip', compression='gzip')


def output_results():

    data = pd.read_csv('../data/all_data.csv', index_col=0)

    complex_features = data[['C', 'H', 'N', 'O', 'S', 'Cl', 'CH3', 'SO3', 'Benzene',
                                'C Chain', 'COOH', 'N Rings', 'O Rings', 'C links']]

    # Compute the correlation matrix
    corr = complex_features.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('../results/heatmap_dyomics_complex.png')
    plt.clf()

    basic_features = data[['C', 'H', 'N', 'O']]
    # Compute the correlation matrix
    corr = basic_features.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('../results/heatmap_dyomics_basic.png')
    plt.clf()

    experiment_results = pd.read_parquet('../results/experiment_results.parquet.gzip')
    metrics = experiment_results[['r2', 'accuracy', 'precision',
                                  'recall', 'f1', 'auc']]
    metrics.to_csv('../results/experiment_results_metrics.csv')
    for index, row in experiment_results.iterrows():
        name = index
        try:
            val_vs_pred = pd.DataFrame({'y_val': list(row['y_val']), 'y_pred': list(row['y_pred'])},
                                       columns=['y_val', 'y_pred'])
            sctplot = sns.scatterplot(x='y_val', y='y_pred', data=val_vs_pred)
            sctplot.plot([0, 300], [0, 300], 'r--', linewidth=1)
            sctplot.set(xlim=(0, 300))
            sctplot.set(ylim=(0, 300))
            plt.savefig('../results/' + name + '_sctplot.png')
            plt.clf()
            y_val = row['y_val'].tolist()
            y_pred = row['y_pred'].tolist()
            df = pd.DataFrame({name + "_y_val": y_val, name + "_y_pred": y_pred})
            df.to_csv(open('../results/' + name + '_sctplot.csv', 'w'))
        except TypeError:
            print("Skipping regression plot for classification model...")

        ax = sns.scatterplot(x='fpr', y='tpr', data=row)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
        ax.set(xlim=(0, 1))
        ax.set(ylim=(0, 1))
        plt.savefig('../results/' + name + '_rocplot.png')
        plt.clf()
        fpr = row['fpr'].tolist()
        tpr = row['tpr'].tolist()
        df = pd.DataFrame({name + "_FPR": fpr, name + "_TPR": tpr})
        df.to_csv(open('../results/' + name + '_rocplot.csv', 'w'))
        '''
        plt.title("Precision-Recall vs Threshold Chart")
        plt.plot(thresholds, precision[: -1], "b--", label="Precision")
        plt.plot(thresholds, recall[: -1], "r--", label="Recall")
        plt.ylabel("Precision, Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="lower left")
        plt.ylim([0, 1])
        '''

def download_data():

    photochemcad_raw_data = pd.read_csv('../raw_data/photochemcad.csv', index_col='Structure')
    photochemcad_compound_names = photochemcad_raw_data.index
    if not path.exists('../data/pubChemCompositionRetrival_temp.csv'):
        compoundRetrival.pubChemCompositionRetrival(photochemcad_compound_names)


if __name__ == "__main__":
    try:
        os.mkdir('../results')
    except OSError as error:
        print(error)
    try:
        os.mkdir('../data')
    except OSError as error:
        print(error)

    download_data()
    cleanupdata()
    run_experiments()
    output_results()
