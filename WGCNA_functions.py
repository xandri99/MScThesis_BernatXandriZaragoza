"""
WGCNA Implementation Using Numpy. 
All methods are commented and callable functions, for an easier use.

"""

################################################################################################
#                                        IMPORTS
################################################################################################

from functools import wraps
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import sys
import gc
import time

from lifelines import KaplanMeierFitter

from statsmodels.formula.api import ols

import scipy.cluster.hierarchy as scipy_hierarchy
from scipy.spatial.distance import squareform
from scipy import stats

from sklearn.decomposition import PCA

################################################################################################





################################################################################################
#                                   CONFIGURATIONS
################################################################################################

# Settings for printing dataframes
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)
DPI_GENERAL = 150


# Colors for the terminal outputs
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

OKBLUE = "\033[94m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"


def measure_time(func):
    """
    A decorator that measures the execution time of a function. It calculates the duration by recording
    the time before and after the function's execution.

    Parameters:
    - func (callable): The function to measure.

    Returns:
    - callable: A wrapper function that, when called, executes the wrapped function, measures its
      execution time, prints the duration, and then returns the function's result.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record the start time of the function execution.

        start_time = time.time()
        result = func(*args, **kwargs)

        # Record the end time and calculates elapsed time.
        end_time = time.time()
        execution_time = end_time - start_time

        # Print the name of the function and its execution time in seconds.
        print(f"\t\tThe function {func.__name__} ran in {execution_time:.2f} seconds.")

        # Return the result of the function execution.
        return result
    return wrapper

################################################################################################





################################################################################################
#                                       PLOTS
################################################################################################

def plot_pca(dataframe, title, ax = None):
    """
    Performs PCA on the provided dataframe and plots the first two principal components for visualization.
    
    Parameters:
    - dataframe (DataFrame): The dataframe to perform PCA on.
    - title (str): The title of the plot.
    
    Returns:
    - None, it generates the plot
    """
    # Perform PCA analysis
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(dataframe.iloc[:, 1:].T)

    # Plot the first two principal components
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')



@measure_time
def plot_heatmap(data, title, figures_dir, vmin, vmax):
    """
    Generates a heatmap of the desired data, and stores it as a file.
    
    Parameters:
    - data (DataFrame, matrix, etc): The data to plot as a heatmap.
    - title (str): The title of the plot.
    
    Returns:
    - None, it generates the plot
    """
    print(f"{BOLD}{OKBLUE}Plotting and Saving {title}...{ENDC}")
    
    # using pcolorfast for speed
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.imshow(data, cmap='coolwarm', vmin = vmin, vmax = vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(cax)

    plt.title(title, fontsize=20)
    plt.xlabel('Genes', fontsize=10)
    plt.ylabel('Genes', fontsize=10)
    plt.savefig(figures_dir + title, dpi=DPI_GENERAL)
    plt.show()
    
    print(f"{BOLD}{OKBLUE}Done{ENDC}")



def printSoftThresholdSearch(results, RsquaredCut, optimal_power, figures_dir):
    """
    Generates and saves plots for the Scale-Free Topology fit analysis and Mean Connectivity 
    as functions of the soft-thresholding power used in network construction. 

    Parameters:
    - results (DataFrame): A DataFrame containing the soft-thresholding powers and their 
      corresponding R² values for scale-free fit and mean connectivity measures.
    - RsquaredCut (float): The R² value used as a cutoff for determining the scale-free topology 
      fit, indicating a satisfactory fit when exceeded by the actual R² values.
    - optimal_power (int): The soft-thresholding power at which the scale-free topology fit 
      R² is above the RsquaredCut and possibly optimal for network construction.
    - figures_dir (str): The directory path where the generated plots will be saved.

    Returns:
    - None, it generates and saves the plots.
    """
    print(f"{BOLD}{OKBLUE}Plotting and Saving Scale-Free Topology fit analysis...{ENDC}")
    title_figure1 = 'Scale-Free Topology Analysis'
    title_figure2 = 'Mean Connectivity'

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plotting Scale-Free Topology Analysis
    axs[0].plot(results.index, results["R²"], marker='o', linestyle='-', label='Scale-Free Topology Fit R^2')
    axs[0].axhline(y=RsquaredCut, color='red', linestyle='--', label=f'R^2 cut-off: {RsquaredCut}')
    axs[0].axvline(x=optimal_power, color='green', linestyle='-', label=f'Optimal Power: {optimal_power}')
    axs[0].set_xlabel('Soft Thresholding Power')
    axs[0].set_ylabel('Scale-Free Topology Fit R^2')
    axs[0].set_title(title_figure1, fontsize=20)
    axs[0].legend()

    # Plotting Mean Connectivity vs. Soft Thresholding Power
    mean_connect_optimal_power = results.loc[optimal_power, 'mean(connectivity)']
    axs[1].plot(results.index, results['mean(connectivity)'], marker='o', linestyle='-', color='blue', label='Mean Connectivity')
    axs[1].axhline(y=mean_connect_optimal_power, color='red', linestyle='--', label=f'Mean Conn at Opt-Power: {mean_connect_optimal_power:.3f}')
    axs[1].axvline(x=optimal_power, color='green', linestyle='--', label=f'Optimal Power: {optimal_power}')
    axs[1].set_xlabel('Soft Thresholding Power')
    axs[1].set_ylabel('Mean Connectivity')
    axs[1].set_title(title_figure2, fontsize=20)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(figures_dir + title_figure1 + " and " + title_figure2, dpi=DPI_GENERAL)
    plt.show()
    print(f"{BOLD}{OKBLUE}Done\n{ENDC}")


def plot_dendogram(data, title, figures_dir):
    """
    Generates a dendrogram to visually represent the hierarchical clustering of data, 
    used here to show the clustering of genes based on their similarity or 
    dissimilarity measured by distance from the Topological Overlap Matrix (TOM).
    
    Parameters:
    - data (ndarray or a linkage matrix): The hierarchical clustering encoded as a linkage 
      matrix or raw data array. If it's a linkage matrix, it describes the hierarchical 
      clustering structure.
    - title (str): The title of the dendrogram plot.
    - figures_dir (str): The directory path where the generated dendrogram will be saved.
    
    Returns:
    - None, it generates and saves the dendrogram plot.
    """
    print(f"{BOLD}{OKBLUE}Plotting and Saving {title}...{ENDC}")

    plt.figure(figsize=(15, 10))
    scipy_hierarchy.dendrogram(data, truncate_mode=None, color_threshold=0)
    plt.title(title, fontsize=20)
    plt.xlabel('Genes', fontsize=10)
    plt.xticks([])
    plt.ylabel('Distance taken from the TOM', fontsize=10)
    plt.savefig(figures_dir + title, dpi=DPI_GENERAL)
    plt.show()
    print(f"{BOLD}{OKBLUE}Done{ENDC}")


def plot_module_distribution(module_assignment):
    """
    Plots the distribution of genes across modules with a structured figure layout:
    a histogram of the number of genes per module on top and two pie charts underneath.

    Parameters:
    - module_assignment (pd.DataFrame): DataFrame containing 'Module' column with module assignments for each gene.
    """
    # Create a figure and a grid of subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Histogram of the number of genes in each module, excluding not assigned (Module 0)
    ax1 = fig.add_subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    genes_in_modules = module_assignment[module_assignment['Module'] != 0]['Module'].value_counts().sort_index()
    ax1.bar(genes_in_modules.index, genes_in_modules.values)
    ax1.set_title('Number of Genes in Each Module')
    ax1.set_xlabel('Module')
    ax1.set_ylabel('Number of Genes')
    
    # Setup for pie charts
    ax2 = fig.add_subplot(2, 2, 3)  # 2 rows, 2 columns, 3rd subplot (bottom left)
    ax3 = fig.add_subplot(2, 2, 4)  # 2 rows, 2 columns, 4th subplot (bottom right)
    
    # Pie chart of genes distribution across modules
    module_sizes = module_assignment['Module'].value_counts()
    ax2.pie(module_sizes, labels=module_sizes.index, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Distribution of Genes Across Modules')
    
    # Percentage of Genes Assigned to Clusters vs Not Assigned
    total_genes = len(module_assignment)
    genes_not_assigned_count = module_assignment[module_assignment['Module'] == 0].shape[0]
    genes_assigned_count = total_genes - genes_not_assigned_count
    sizes = [genes_assigned_count, genes_not_assigned_count]
    labels = ['Assigned to Modules', 'Not Assigned (Module 0)']
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax3.set_title('Percentage of Genes Assigned to Clusters vs Not Assigned')

    plt.tight_layout()
    plt.show()


def correlation_pvalue_heatmap(correlations, p_values, figures_dir, title=""):
    """
    Generates a heatmap showing correlations between module eigengenes and clinical traits, 
    with annotations for the corresponding p-values.
    
    Parameters:
    - correlations (DataFrame): A square DataFrame where rows and columns represent modules 
      and clinical traits respectively, and the cell values are the correlations between them.
    - p_values (DataFrame): A DataFrame of the same shape as correlations, where each cell 
      contains the p-value associated with the corresponding correlation.
    - figures_dir (str): Directory path where the generated heatmap will be saved.
    
    Returns:
    - None, it generates and saves the plot as a file.
    """
    print(f"{BOLD}{OKBLUE}Plotting and Saving the Module EigenGene to Clinical Trait Correlation...{ENDC}")
    title_figure = 'Module Eigengene to Clinical Trait Correlation (' + title + ')'

    annotations = correlations.round(3).astype(str) + '\n(' + p_values.round(5).astype(str) + ')'

    plt.figure(figsize=(40, 40)) 
    sns.heatmap(correlations, annot=annotations.values, fmt='', cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title(title_figure, fontsize=20)
    plt.xlabel('Selected Clincal Traits', fontsize=10)
    plt.ylabel('Identified Modules, represented by their EigenGene', fontsize=10)
    plt.savefig(figures_dir + title_figure, dpi=150)
    plt.show()
    print(f"{BOLD}{OKBLUE}Done{ENDC}")

    


def correlation_heatmap(correlations, p_values, figures_dir, title="", p_value_th=0.45):
    """
    Generates a heatmap showing correlations between module eigengenes and clinical traits, 
    with annotations for the corresponding p-values.
    
    Parameters:
    - correlations (DataFrame): A square DataFrame where rows and columns represent modules 
      and clinical traits respectively, and the cell values are the correlations between them.
    - p_values (DataFrame): A DataFrame of the same shape as correlations, where each cell 
      contains the p-value associated with the corresponding correlation.
    - figures_dir (str): Directory path where the generated heatmap will be saved.
    - p_value_th (float): p-value threshold. Over this, cells don't get colored
    
    Returns:
    - None, it generates and saves the plot as a file.
    """
    print(f"{BOLD}{OKBLUE}Plotting and Saving the Module EigenGene to Clinical Trait Correlation...{ENDC}")
    title_figure = 'Module Eigengene to Clinical Trait Correlation (' + title + ')'
    
    # Pvalue and Corr annotations
    annotations = correlations.round(3).astype(str)

    # Custom colors for under p_value
    pval_filtered = p_values <= p_value_th

    # Plot
    plt.figure(figsize=(40, 40)) 
    ax = sns.heatmap(correlations, annot=annotations.values, fmt='', cmap='coolwarm', center=0, vmin=-1, vmax=1, mask=~pval_filtered)
    # Samples under 0.05 p value
    sns.heatmap(correlations, mask=pval_filtered, cmap=['#ebebeb'], cbar=False, annot=False, ax=ax)
    legend_patch = mpatches.Patch(color='#ebebeb', label=f'p-value > {p_value_th}')
    plt.legend(handles=[legend_patch], loc='upper right', fontsize=20)

    plt.title(title_figure, fontsize=20)
    plt.xlabel('Selected Clincal Traits', fontsize=10)
    plt.ylabel('Identified Modules, represented by their EigenGene', fontsize=10)
    plt.savefig(figures_dir + title_figure, dpi=150)
    plt.show()
    print(f"{BOLD}{OKBLUE}Done{ENDC}")



def correlation_heatmap_paper(correlations, p_values, figures_dir, title="", p_value_th=0.45):
    """
    Generates a heatmap showing correlations between module for report
    """
    print(f"{BOLD}{OKBLUE}Plotting and Saving the Module EigenGene to Clinical Trait Correlation...{ENDC}")
    title_figure = 'Module Eigengene to Clinical Trait Correlation (' + title + ')'
    

    # Custom colors for under p_value
    pval_filtered = p_values <= p_value_th

    # Plot
    plt.figure(figsize=(40, 40)) 
    ax = sns.heatmap(correlations, fmt='', cmap='coolwarm', center=0, vmin=-1, vmax=1, mask=~pval_filtered)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)



    # Samples under 0.05 p value
    sns.heatmap(correlations, mask=pval_filtered, cmap=['#ebebeb'], cbar=False, annot=False, ax=ax)

    plt.savefig(figures_dir + title_figure, dpi=150)
    plt.show()
    print(f"{BOLD}{OKBLUE}Done{ENDC}")

    


def correlation_heatmap_small(correlations, p_values, figures_dir, title="", p_value_th=0.45):
    """
    """
    print(f"{BOLD}{OKBLUE}Plotting and Saving the Module EigenGene to Clinical Trait Correlation...{ENDC}")
    
    # Pvalue and Corr annotations
    annotations = correlations.round(3).astype(str)

    # Custom colors for under p_value
    pval_filtered = p_values <= p_value_th

    # Plot
    plt.figure(figsize=(20, 20)) 
    ax = sns.heatmap(correlations, annot=annotations.values, fmt='', cmap='coolwarm', center=0, vmin=-1, vmax=1, mask=~pval_filtered, annot_kws={"size": 12})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)

    # Samples under 0.05 p value
    sns.heatmap(correlations, mask=pval_filtered, cmap=['#ebebeb'], cbar=False, annot=False, ax=ax)

    plt.show()
    print(f"{BOLD}{OKBLUE}Done{ENDC}")



################################################################################################






################################################################################################
#                                       PREPROCESSING
################################################################################################

@measure_time
def simple_preprocess(raw_data):
    """
    Simple initial preprocessing on raw data by removing genes with no variation across samples.
    
    Parameters:
    - raw_data (DataFrame): The raw data as a pandas DataFrame.
    
    Returns:
    - DataFrame: The cleaned dataset with genes having no variation removed.
    """
    print(f"{BOLD}{OKBLUE}Pre-processing...{ENDC}")

    # Remove genes with no variation across samples
    cleaned_dataset = raw_data.loc[:, (raw_data != 0).any(axis=0)]

    # Print the number of genes removed
    num_genes_removed = raw_data.shape[1] - cleaned_dataset.shape[1]
    print(f"{BOLD}{WARNING}simple_preprocess function removed {num_genes_removed} genes{ENDC}")
    
    print(f"{BOLD}{OKBLUE}Done...{ENDC}")
    return cleaned_dataset


@measure_time
def preprocess_TPM(raw_data, expression_th):
    """
    Preprocesses raw data by filtering out low expression genes and applying log transformation.
    
    Parameters:
    - raw_data (DataFrame): The raw data as a pandas DataFrame.
    - expression_th (int): The value of expression under which genes are eliminated.
    
    Returns:
    - DataFrame: The cleaned and transformed dataset.
    """
    print(f"{BOLD}{OKBLUE}Pre-processing...{ENDC}")

    # Filter out genes with low expression across all samples
    cleaned_dataset = raw_data.loc[:, (raw_data > expression_th).any(axis=0)].copy()
    
    # Apply log2 transformation to all values except for the first column (gene identifiers)
    cleaned_dataset.iloc[:, 1:] = np.log2(cleaned_dataset.iloc[:, 1:] + 1)

    # Print the number of genes removed
    num_genes_removed = raw_data.shape[1] - cleaned_dataset.shape[1]
    print(f"{BOLD}{WARNING}preprocess_TPM function removed {num_genes_removed} genes{ENDC}")
    
    print(f"{BOLD}{OKBLUE}Done...{ENDC}")
    return cleaned_dataset


@measure_time
def preprocess_TPM_outlier_deletion(raw_data, expression_th, trait_dataset):
    """
    Cleans raw data by filtering out low expression genes, applying log transformation, and removing outliers based on PCA analysis.
    
    Parameters:
    - raw_data (DataFrame): The raw data as a pandas DataFrame.
    - expression_th (int): The value of expression under which genes are eliminated.
    
    Returns:
    - DataFrame: The dataset after preprocessing and outlier removal.
    """
    print(f"{BOLD}{OKBLUE}Pre-processing...{ENDC}")

    # Filter out genes with low expression across all samples
    cleaned_dataset = raw_data.loc[:, (raw_data > expression_th).any(axis=0)].copy()
    
    # Apply log2 transformation to all values except for the first column (gene identifiers)
    cleaned_dataset.iloc[:, 1:] = np.log2(cleaned_dataset.iloc[:, 1:] + 1)
    
    # Outlier detection and removal based on PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(cleaned_dataset.iloc[:, 1:])  # NOT Transpose to have samples as rows for PCA
    z_scores = np.abs(stats.zscore(pca_result, axis=0))
    good_samples = (z_scores < 3).all(axis=1)                      # Keeping samples within 3 standard deviations
    cleaned_dataset = cleaned_dataset[good_samples]

    # Data Standardization (Z-score normalization)
    cleaned_dataset.iloc[:, 1:] = cleaned_dataset.iloc[:, 1:].apply(stats.zscore, axis=0)

    # Print the number of genes removed
    num_genes_removed = raw_data.shape[1] - cleaned_dataset.shape[1]
    print(f"{BOLD}{WARNING}preprocess_TPM_outlier_deletion function removed {num_genes_removed} genes{ENDC}")

    # Print the number of genes removed
    num_pacients_removed = raw_data.shape[0] - cleaned_dataset.shape[0]
    print(f"{BOLD}{WARNING}preprocess_TPM_outlier_deletion function removed {num_pacients_removed} pacients{ENDC}")

    # Adjust the traits dataset to match the new list of pacients
    trait_dataset_filtered = trait_dataset.loc[trait_dataset.index.isin(cleaned_dataset.index)]

    print(f"{BOLD}{OKBLUE}Done...{ENDC}")
    return cleaned_dataset, trait_dataset_filtered


@measure_time
def preprocess_TPM_Zscore(raw_data, expression_th):
    """
    Cleans raw data by filtering out low expression genes, applying log transformation, and Zscore normalization.
    
    Parameters:
    - raw_data (DataFrame): The raw data as a pandas DataFrame.
    - expression_th (int): The value of expression under which genes are eliminated.
    
    Returns:
    - DataFrame: The dataset after preprocessing and outlier removal.
    """
    print(f"{BOLD}{OKBLUE}Pre-processing...{ENDC}")

    # Filter out genes with low expression across all samples
    cleaned_dataset = raw_data.loc[:, (raw_data > expression_th).any(axis=0)].copy()
    
    # Apply log2 transformation to all values except for the first column (gene identifiers)
    cleaned_dataset.iloc[:, 1:] = np.log2(cleaned_dataset.iloc[:, 1:] + 1)



    # Data Standardization (Z-score normalization)
    cleaned_dataset.iloc[:, 1:] = cleaned_dataset.iloc[:, 1:].apply(stats.zscore, axis=0)



    # Print the number of genes removed
    num_genes_removed = raw_data.shape[1] - cleaned_dataset.shape[1]
    print(f"{BOLD}{WARNING}preprocess_TPM_outlier_deletion function removed {num_genes_removed} genes{ENDC}")

    # Print the number of genes removed
    num_pacients_removed = raw_data.shape[0] - cleaned_dataset.shape[0]
    print(f"{BOLD}{WARNING}preprocess_TPM_outlier_deletion function removed {num_pacients_removed} pacients{ENDC}")

    # Adjust the traits dataset to match the new list of pacients

    print(f"{BOLD}{OKBLUE}Done...{ENDC}")
    return cleaned_dataset


@measure_time
def preprocess_agresive(raw_data):
    """
    From GCNN paper.
    """
    print(f"{BOLD}{OKBLUE}Pre-processing...{ENDC}")

    
    ###Important to notice than in the paper, they work with log2(FPKM + 1) and we have TPM
    # Applying log transformation to TPM data
    transcriptomics_log_transformed = np.log2(raw_data + 1)


    # thresholds as per the paper
    mean_threshold = 0.5
    std_dev_threshold = 0.8

    # Kepp only genes with mean expression>0.5 and standard deviation>0.8
    genes_mean_expression = transcriptomics_log_transformed.mean(axis=0)
    genes_std_deviation = transcriptomics_log_transformed.std(axis=0)
    transcriptomics_clean = transcriptomics_log_transformed.loc[:, (genes_mean_expression > mean_threshold) & (genes_std_deviation > std_dev_threshold)]

    # Print the number of genes removed
    num_genes_removed = raw_data.shape[1] - transcriptomics_clean.shape[1]
    print("Preprocessing removed", str(num_genes_removed), "genes")
    print("We have", str(transcriptomics_clean.shape[1]), "genes left")


    # Normalize gene expression between 0 and 1
    transcriptomics_normalized = (transcriptomics_clean - transcriptomics_clean.min()) / (transcriptomics_clean.max() - transcriptomics_clean.min())
    

    print(f"{BOLD}{OKBLUE}Done...{ENDC}")
    return transcriptomics_normalized


################################################################################################





################################################################################################
#                                     CORRELATION MATRIX
################################################################################################

@measure_time 
def correlation_matrix(dataset, want_plots, figures_dir):
    """
    The Correlation matrix, also known as Co-expression Matrix or Similarity matrix is calculated by
    calculating the correlation between the expression profiles of all genes. The expression profile of 
    a gene is a vector containing the expression levels of that gene for all samples (pacients)
    
    Parameters:
    - dataset (DataFrame): The raw data as a pandas DataFrame.
    
    Returns:
    - DataFrame: The cleaned dataset with genes having no variation removed.
    """
    print(f"{BOLD}{OKBLUE}Calculating Correlation Matrix...{ENDC}")

    # Calculate the correlation matrix using Pearson correlation
    np_transcriptomics_dataset = dataset.T.to_numpy()
    correlation_matrix_np = np.corrcoef(np_transcriptomics_dataset)

    # Enforce ranges of values and diagonal in matrix 
    correlation_matrix_np = np.clip(correlation_matrix_np, -1.0, 1.0)
    np.fill_diagonal(correlation_matrix_np, 1)
    
    print(f"{BOLD}{OKBLUE}Done...{ENDC}")

    if want_plots:
        title_figure = 'Gene Expression Correlation Matrix Heatmap'
        plot_heatmap(correlation_matrix_np, title_figure, figures_dir, vmin = -1, vmax = 1)

    return correlation_matrix_np


def matrix_np_check(np_matrix, max_value, min_value, expected_diag_value):
    """
    Checks the validity of a NumPy matrix by ensuring all values are within a specified range
    and the diagonal elements meet the expected value. It is useful for verifying the integrity
    of matrices where specific value constraints are critical, such as correlation matrices.

    Parameters:
    - np_matrix (numpy.ndarray): The matrix to be checked as a NumPy ndarray.
    - max_value (float): The maximum value allowed in the matrix.
    - min_value (float): The minimum value allowed in the matrix.
    - expected_diag_value (float): The expected value along the diagonal of the matrix.

    Outputs:
    - Prints a message indicating whether the matrix satisfies all conditions or reports
      the largest and smallest values found if any conditions are not met.
    """
    # Check for values outside the allowed range
    smaller_values_check = not((np_matrix < min_value).any().any())
    bigger_values_check = not((np_matrix > max_value).any().any())

    if smaller_values_check and bigger_values_check:
        # Check for values in diagonal
        if np.all(np.diag(np_matrix) == expected_diag_value):
            print(f"{BOLD}{OKBLUE}The matrix satisfies all restrictions.{ENDC}")
    else:
        # For visual inspection print the biggest and smallest value in the Dataframe
        print(f'The biggest value in this matrix is: {np_matrix.max().max()}\
        \nThe smallest value in this matrix is: {np_matrix.min().min()}')
        print('\n')


################################################################################################





################################################################################################
#                                     ADJACENCY MATRIX
################################################################################################

def scaleFreeFitIndex(connectivity, block_size):
    """
    Evaluates the fit of a network's topology to a scale-free model (power-law distribution) by analyzing 
    its connectivity.
    The function calculates the R-squared value of a linear regression model fitted to the logarithm
    of the mean connectivity and the logarithm of the probability density per block of connectivity values.

    Parameters:
    - connectivity (array-like): The connectivity data of the network.
    - block_size (int): The number of bins to divide the connectivity data into.

    Returns:
    - pd.DataFrame: A DataFrame containing R-squared, slope of the linear regression, and adjusted R-squared.
    """
    # Filter out zero connectivity values
    connectivity = connectivity[connectivity > 0]

    # Create a DataFrame and discretize connectivity into specified number of blocks
    connectivity = pd.DataFrame({'data': connectivity})
    connectivity['discretized_connectivity'] = pd.cut(connectivity['data'], block_size)

    # Compute mean connectivity and probability density for each block
    per_block_stats = connectivity.groupby('discretized_connectivity', observed=False)['data']\
                                    .agg(['mean', 'count']) \
                                    .reset_index()\
                                    .rename(columns={'mean': 'mean_connectivity_per_block', 'count': 'count_per_block'})
    per_block_stats['probability_density_per_block'] = per_block_stats['count_per_block'] / len(connectivity)

    # Impute missing values in blocks with their midpoint values
    breaks = np.linspace(start=connectivity['data'].min(), stop=connectivity['data'].max(), num=block_size + 1)
    mid_points_blocks = 0.5 * (breaks[:-1] + breaks[1:])  # Mid-points of blocks
    for i, row in per_block_stats.iterrows():
        if pd.isnull(row['mean_connectivity_per_block']) or row['mean_connectivity_per_block'] == 0:
            per_block_stats.at[i, 'mean_connectivity_per_block'] = mid_points_blocks[i]

    # Perform logarithmic transformation for linear regression analysis
    per_block_stats['log_mean_conn'] = np.log10(per_block_stats['mean_connectivity_per_block'])
    per_block_stats['log_prob_distr_conn'] = np.log10(per_block_stats['probability_density_per_block'] + 1e-9)

    # Linear regression model for R-squared
    simple_linear_regression_model = ols('log_prob_distr_conn ~ log_mean_conn', data=per_block_stats).fit()
    rsquared = simple_linear_regression_model.rsquared
    slope = simple_linear_regression_model.params['log_mean_conn']
    
    # Quadratic Regression Model for Adjusted R-squared
    quadratic_regression_model = ols('log_prob_distr_conn ~ log_mean_conn + I(log_mean_conn**2)', data=per_block_stats).fit()
    rsquared_adj = quadratic_regression_model.rsquared_adj
    
    # Return the fitting statistics
    return pd.DataFrame({
        'Rsquared.SFT': [rsquared],
        'slope.SFT': [slope],
        'Rsquared Adjusted': [rsquared_adj]
    })


@measure_time
def pickSoftThreshold(correlation_matrix_np, transcriptomics_dataset_filtered, RsquaredCut, MeanCut, want_plots, figures_dir, block_size):
    """
    Analyzes scale-free topology for multiple soft thresholding powers.
    Soft power-thresholding is a value used to power each value of the correlation matrix of the genes to that threshold.
    The assumption is that by raising the correlation values to a power, we will reduce the noise of the correlations in
    the adjacency matrix, therefore putting in relevance important links and tuning down the noise.

    To pick up the threshold, the pickSoftThreshold function calculates for each possible power if the network resembles
    a scale-free network topology (following a power-law distribution).
    The power which produce a higher similarity with a scale-free network is the one returned.

    This is critical in WGCNA, as the premise of the method is that biological networks often exhibit scale-free properties, 
    meaning that a few nodes (genes) are highly connected, while most have few connections.

    Parameters:
    - correlation_matrix (array-like): The correlation matrix of gene expressions.
    - RsquaredCut (float): Minimum R-squared value for a power to be considered valid.
    - MeanCut (float): Maximum mean connectivity allowed for a power to be considered valid.

    Returns:
    - tuple: A tuple containing the optimal power and a DataFrame with the evaluation results for all tested powers.
    """
    # Turn correlation matrix into pandas for row computations
    correlation_matrix = pd.DataFrame(correlation_matrix_np, columns=transcriptomics_dataset_filtered.columns, index=transcriptomics_dataset_filtered.columns)

    # Define the range of powers to evaluate
    powerVector = list(range(1, 11)) + list(range(12, 21, 2))

    # Initialize a DataFrame to store the results
    results = pd.DataFrame(index=powerVector, columns=["Power"])

    # Evaluate each power
    for power in powerVector:
        # Calculate adjacency matrix from the correlation_matrix with each power
        adjacency_matrix = np.power(np.abs(correlation_matrix), power)
        
        # Calculate connectivity for each node/gene
        connectivity = adjacency_matrix.sum(axis=0) - 1  # we remove the autocorrelation for each row
        
        # Assess fit to a scale-free topology
        fit_values = scaleFreeFitIndex(connectivity, block_size)
        
        # Store results
        results.loc[power, "Power"] = power
        results.loc[power, "R²"] = fit_values['Rsquared.SFT'].values[0]
        results.loc[power, "Slope"] = fit_values['slope.SFT'].values[0]
        results.loc[power, "Exponential R² Adjusted"] = fit_values['Rsquared Adjusted'].values[0]
        results.loc[power, "mean(connectivity)"] = connectivity.mean()
        results.loc[power, "median(connectivity)"] = connectivity.median()
        results.loc[power, "max(connectivity)"] = connectivity.max()
    
    print(results)


    # Determine the optimal power based on the specified criteria
    valid_powers = results[(results["R²"] > RsquaredCut) & (results["mean(connectivity)"] < MeanCut)]
    if not valid_powers.empty:
        optimal_power = valid_powers.index[0]
    else:
        optimal_power = results["R²"].idxmax()


    print(f"{BOLD}{OKGREEN}The optimal Power-Threshold found is {optimal_power}.{ENDC}")

    if want_plots:
        printSoftThresholdSearch(results, RsquaredCut, optimal_power, figures_dir)


    return optimal_power


@measure_time
def adjacencyM_from_correlationM(correlation_matrix_np, optimal_power, adjacency_type, want_plots, figures_dir):
    """
    Converts a correlation matrix into an adjacency matrix using the optimal soft-thresholding power.
    Applies a soft-thresholding power to the absolute values of the correlation matrix
    to create an adjacency matrix, which can be either unsigned or signed. The function also ensures
    that all values in the adjacency matrix are within the [0, 1] range and sets the diagonal to 1.

    Parameters:
    - correlation_matrix_np (numpy.ndarray): The correlation matrix to be converted.
    - optimal_power (float): The power to apply for soft-thresholding.
    - adjacency_type (str): The type of adjacency matrix to create ('unsigned' or 'signed').

    Returns:
    - numpy.ndarray: The adjacency matrix derived from the correlation matrix.
    """
    print(f"{BOLD}{OKBLUE}Calculating Adjacency Matrix...{ENDC}")

    if adjacency_type == "unsigned":
        # Apply soft-thresholding power to the absolute values of the correlation matrix.
        adjacency_matrix_np = np.power(np.abs(correlation_matrix_np), optimal_power)
    elif adjacency_type == 'signed':
        # Convert correlation values to [0, 1] range and apply soft-thresholding power.
        correlation_matrix_np = (correlation_matrix_np + 1) / 2
        adjacency_matrix_np = np.power(correlation_matrix_np, optimal_power)

    # Enforce ranges of values and diagonal in matrix
    adjacency_matrix_np = np.clip(adjacency_matrix_np, 0.0, 1.0)
    np.fill_diagonal(adjacency_matrix_np, 1)
    
    print(f"{BOLD}{OKBLUE}Done\n\n{ENDC}")

    if want_plots:
        title_figure = 'Adjacency Matrix Heatmap'
        plot_heatmap(adjacency_matrix_np, title_figure, figures_dir, vmin = 0, vmax = 1)

    return adjacency_matrix_np


################################################################################################





################################################################################################
#                                   TOPOLOGICAL OVERLAP MATRIX
################################################################################################

@measure_time
def calculate_tom(adjacency_matrix, TOMDenom, adjacency_type, want_plots, figures_dir):
    '''
    Calculates the Topological Overlap Matrix (TOM) for a given adjacency matrix.

    The TOM is a similarity matrix measuring the overlap of shared neighbors between nodes
    (e.g., genes) in a network. 

    Parameters:
    - adjacency_matrix (numpy.ndarray): The numpy adjacency matrix of the network.
    - TOMDenom (str): Specifies the TOM variant to use. Options are "min" for the standard TOM
                      as described by Zhang and Horvath (2005), and "mean" for an experimental variant
                      where the denominator uses the mean instead of the minimum.
    - adjacency_type (str): The type of adjacency matrix to create ('unsigned' or 'signed').

    Returns:
    - numpy.ndarray: The computed Topological Overlap Matrix.
    '''
    print(f"{BOLD}{OKBLUE}Calculating the TOM...{ENDC}")

    # Turn the Adjacency matrix diagonl to 0 for calculations
    np.fill_diagonal(adjacency_matrix, 0)

    # Calculate numerator as A^2
    numerator = np.dot(adjacency_matrix, adjacency_matrix)

    # Compute the sum of connections for each node (row-wise and column-wise)
    row_sum = adjacency_matrix.sum(axis=1)
    col_sum = adjacency_matrix.sum(axis=0)

    # Calculations deppending on the selectod method
    if TOMDenom == 'min':
        denominator = np.minimum.outer(row_sum, col_sum)
    elif TOMDenom == 'mean':
        denominator = (np.outer(row_sum, np.ones_like(row_sum)) + np.outer(np.ones_like(col_sum), col_sum)) / 2

    # Numerator adjustment for unsigned and signed topologies
    if adjacency_type == 'unsigned':
        tom = (numerator + adjacency_matrix) / (denominator + 1 - adjacency_matrix)
    elif adjacency_type == 'signed':
        tom = np.abs(numerator + adjacency_matrix) / (denominator + 1 - np.abs(adjacency_matrix))

    # Set diagonal to 1 as per TOM definition
    np.fill_diagonal(tom, 1)  

    # Handle NaN values and set them to zero
    tom = np.nan_to_num(tom, nan=0)
    
    print(f"{BOLD}{OKBLUE}Done...{ENDC}")

    if want_plots:
        title_figure = 'Topological Overlap Matrix (TOM) Heatmap'
        plot_heatmap(tom, title_figure, figures_dir, vmin = 0, vmax = 1)


    return tom


################################################################################################





################################################################################################
#                                     HIERARCHICAL CLUSTERING
################################################################################################

@measure_time
def hierarchical_clustering(dissTOM_np, want_plots, figures_dir):
    """
    Performs hierarchical clustering on a dissimilarity matrix and optionally generates a dendrogram.

    The function first converts the dissimilarity matrix (1 - TOM) to a condensed form since hierarchical clustering
    in SciPy requires a condensed distance matrix. It then performs clustering using the specified method and optionally
    generates a dendrogram plot of the clustering result.

    Parameters:
    - dissTOM_np (numpy.ndarray): The dissimilarity matrix (1 - TOM) to cluster.
    - want_plots (bool, optional): Whether to generate and save a dendrogram plot. Defaults to False.
    - figures_dir (str, optional): Directory where to save the dendrogram plot. Required if want_plots is True.

    Returns:
    - numpy.ndarray: The linkage matrix from hierarchical clustering.
    """
    print(f"{BOLD}{OKBLUE}Doing Hierarchical clustering over the dissimilarity TOM (1-TOM)...{ENDC}")

    # Convert the square symmetric matrix to condensed form as required by the scipy linkage function
    condensed_dissTOM = squareform(dissTOM_np, checks=False)

    # Perform hierarchical clustering using the specified method
    method = "average"  
    linkage_matrix = scipy_hierarchy.linkage(condensed_dissTOM, method=method)

    print(f"{BOLD}{OKBLUE}Done...\n\n{ENDC}")

    # Print results
    if want_plots:
        title_figure = 'Dendogram from the Hierarchical clustering'
        plot_dendogram(linkage_matrix, title_figure, figures_dir)
    
    return linkage_matrix


################################################################################################





################################################################################################
#                                     MODULE IDENTIFICATION
################################################################################################

@measure_time
def identify_modules_simple_version(linkage_matrix, height_percentile, min_memb_cluster):
    """
    Identifies gene modules from a hierarchical clustering dendrogram by cutting it at a dynamically determined height.
    Small clusters are merged into a single 'no cluster' module. The function aims for a balance in cluster sizes.

    Parameters:
    - linkage_matrix (numpy.ndarray): The linkage matrix representing the dendrogram.
    - height_percentile (float): The percentile of the dendrogram height to use for cutting the tree. 
                                 Values between 0 and 100.
    - min_memb_cluster (int): The minimum number of members a cluster must have to be considered a separate module.

    Returns:
    - pd.DataFrame: A DataFrame mapping each gene to its module.
    - float: The height at which the dendrogram was cut to form clusters.
    """
    print(f"{BOLD}{OKBLUE}Finding Modules from the Dendogram with a Tree-Cutting Algorithm...{ENDC}")

    # Determine the cut height based on the specified percentile of the linkage matrix heights
    cut_height_percentile = np.percentile(linkage_matrix[:, 2], height_percentile)
    
    # Calculate the cut height dynamically based on the dendrogram's maximum height and sensitivity
    max_dendro_height = np.max(linkage_matrix[:, 2])
    sensitivity = max_dendro_height / cut_height_percentile
    cut_height = max_dendro_height / sensitivity

    # Form flat clusters from the dendogram
    cluster_labels = scipy_hierarchy.fcluster(linkage_matrix, t=cut_height, criterion='distance')

    # Map genes to their cluster labels in a dataframe
    module_assignment = pd.DataFrame({'Gene': range(1, len(cluster_labels) + 1), 'Module': cluster_labels})

    # Filter out small modules and join them to the Module 0 (no cluster)
    module_sizes = module_assignment['Module'].value_counts()
    small_modules = module_sizes[module_sizes < min_memb_cluster].index
    module_assignment['Module'] = module_assignment['Module'].apply(lambda x: 0 if x in small_modules else x)

    # Reassign module labels to be consecutive for non-zero modules
    non_zero_modules = module_assignment[module_assignment['Module'] != 0]['Module']
    unique_non_zero_modules = pd.Categorical(non_zero_modules).codes + 1
    module_assignment.loc[module_assignment['Module'] != 0, 'Module'] = unique_non_zero_modules

    return module_assignment, cut_height



@measure_time
def find_optimal_cut_height(linkage_matrix):
    """
    Determines an optimal cut height for hierarchical clustering by analyzing
    the distribution of linkage distances and identifying an 'elbow' point.

    Parameters:
    - linkage_matrix (numpy.ndarray): The linkage matrix representing the dendrogram.

    Returns:
    - float: An optimal cut height derived from the linkage matrix.
    """
    # Derive the linkage distances
    distances = linkage_matrix[:, 2]
    distances_sorted = np.sort(distances)
    
    # Calculate the gradient of distances
    gradient = np.diff(distances_sorted)
    
    # Identify the elbow point as the maximum gradient
    elbow_index = np.argmax(gradient)
    optimal_cut_height = distances_sorted[elbow_index]

    return optimal_cut_height



@measure_time
def identify_modules_auto_deep_split(linkage_matrix, dist_matrix, min_memb_cluster):
    """
    Identifies modules by hierarchical clustering, enhancing cluster identification with gap statistics
    and handling of small clusters and singletons.

    Parameters:
    - linkage_matrix (numpy.ndarray): The hierarchical clustering linkage matrix.
    - dist_matrix (numpy.ndarray): The pairwise distance matrix of the items being clustered. dissTOM
    - min_memb_cluster (int): Minimum number of members to consider a cluster valid.

    Returns:
    - pd.DataFrame: Module assignments for each gene/item.
    - float: The height used to cut the dendrogram to form initial clusters.
    """
    
    # Determine the cut height from the dendrogram
    cutHeight = find_optimal_cut_height(linkage_matrix)

    # Assign clusters based on the cut height
    cluster_labels = scipy_hierarchy.fcluster(linkage_matrix, t=cutHeight, criterion='distance')
    
    # Calculate core scatter and gap for each cluster
    n_clusters = np.max(cluster_labels)
    core_scatters = []
    gaps = []
    
    for i in range(1, n_clusters + 1):
        cluster_mask = cluster_labels == i
        within_cluster_distances = dist_matrix[np.ix_(cluster_mask, cluster_mask)]
        between_cluster_distances = dist_matrix[np.ix_(cluster_mask, ~cluster_mask)]
        
        core_scatter = np.mean(within_cluster_distances[np.triu_indices_from(within_cluster_distances, k=1)])
        core_scatters.append(core_scatter)
        
        gap = np.min(between_cluster_distances) if between_cluster_distances.size > 0 else np.inf
        gaps.append(gap)
    
    cluster_metrics = pd.DataFrame({'Cluster': range(1, n_clusters + 1), 'CoreScatter': core_scatters, 'Gap': gaps})
    
    # Handling Small Clusters and Singletons
    module_assignment = pd.DataFrame({'Item': range(len(cluster_labels)), 'Module': cluster_labels})
    cluster_sizes = module_assignment['Module'].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_memb_cluster].index
    
    for small_cluster in small_clusters:
        members_idx = module_assignment[module_assignment['Module'] == small_cluster].index
        
        if len(members_idx) == 1:  # Handle singleton
            item_idx = members_idx[0]
            distances = dist_matrix[item_idx]
            distances[item_idx] = np.inf  # Ignore distance to itself
            closest_item_idx = np.argmin(distances)
            closest_cluster = module_assignment.iloc[closest_item_idx]['Module']
            module_assignment.at[item_idx, 'Module'] = closest_cluster
        else:
            # Additional logic for handling small clusters can be implemented here
            pass

    # Optionally, recompute cluster labels to be consecutive
    module_assignment['Module'] = pd.factorize(module_assignment['Module'])[0] + 1

    return module_assignment, cutHeight


################################################################################################





################################################################################################
#                                         EIGENGENE
################################################################################################

def expression_profile_for_cluster(module_assignment, transcriptomics_dataset_filtered):
    """
    Merges module assignments with gene expression data to create a combined dataset.

    Parameters:
    - module_assignment (pd.DataFrame): DataFrame containing 'Gene Name' and 'Module' columns.
    - transcriptomics_dataset_filtered (pd.DataFrame): Filtered gene expression dataset with genes as rows.

    Returns:
    - pd.DataFrame: Merged DataFrame containing module assignments and expression data for each gene.
    """
    # Transpose the dataset to have genes as columns
    expression_profile = transcriptomics_dataset_filtered.T
    expression_profile = expression_profile.reset_index()

    # Merge the module assignments with the expression data based on gene names
    merged_df = pd.merge(module_assignment, expression_profile, left_on='Gene Name', right_on='index', how='inner')

    # Drop unnecessary columns
    merged_df = merged_df.drop('index', axis=1)
    merged_df = merged_df.drop('Gene', axis=1)

    return merged_df


@measure_time
def calculate_eigen_genes(expression_profiles, want_plots, figures_dir):
    """
    Calculates the eigengenes for each module in the expression profiles.

    Parameters:
    - expression_profiles (pd.DataFrame): DataFrame with gene expression profiles, including 'Module' assignments.

    Returns:
    - pd.DataFrame: A DataFrame of eigengenes for each module.
    """
    print(f"{BOLD}{OKBLUE}Calculating EigenGenes...{ENDC}")

    eigengenes = []
    
    ## Iterate through each module to calculate its eigen gene
    for module in expression_profiles['Module'].unique():

        # Skip module 0 as it represents unassigned genes
        if module == 0:
            continue
        else:
            # Extract the Expression Profile for all genes in this module
            module_expression_profile = expression_profiles[expression_profiles['Module'] == module].iloc[:, 2:]  # Exclude Gene Name and Module columns

            # Perform PCA on the expression data of the current module
            pca = PCA(n_components=1)
            pca.fit(module_expression_profile)
            
            # The first principal component is the eigengene
            eigengene = pca.components_[0]

            # Create a DataFrame for the eigengene with the correct sample labels and the module id
            eigengene_df = pd.DataFrame(eigengene.reshape(1, -1), columns=expression_profiles.columns[2:])
            eigengene_df.insert(0, 'Module', module)
            eigengenes.append(eigengene_df)
    
    eigengenes = pd.concat(eigengenes, ignore_index=True)
    
    print(f"{BOLD}{OKBLUE}Done\n\n{ENDC}")    


    ## Plot the Expression Profile for the Eigengenes across pacients
    if want_plots:
        print(f"{BOLD}{OKBLUE}Plotting and Saving the Eigengene Expression Profile Across Samples...{ENDC}")
        title_figure = 'Eigengene Expression Profile Across Samples'

        sample_labels = eigengenes.columns[1:]

        plt.figure(figsize=(15, 10))
        for index, row in eigengenes.iterrows():
            # Convert eigengene array stored as list back to numpy array for plotting
            eigengene_values = np.array(row[1:].values)
            
            # Plotting the eigengene values
            plt.plot(sample_labels, eigengene_values, label=f'Module {row["Module"]}')

        plt.title(title_figure, fontsize=20)
        plt.xlabel('Samples (pacients)', fontsize=10)
        plt.ylabel('Eigengene Expression Level', fontsize=10)
        plt.xticks(rotation=90)
        plt.xticks([])
        #plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir + title_figure, dpi=DPI_GENERAL)
        plt.show()
        print(f"{BOLD}{OKBLUE}Done{ENDC}")
    
    return eigengenes


################################################################################################





################################################################################################
#                                    MODULE-TRAIT RELATIONSHIP
################################################################################################

def encode_categorical(df, column):
    """
    Encodes a categorical column in a DataFrame. 

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to be encoded.
    - column (str): The name of the column to encode.

    Returns:
    - pd.DataFrame: The DataFrame with the specified column encoded. 
    """
    if df[column].dtype == 'object':
        if column == 'Age':  # Assuming Age has an order
            df[column] = df[column].str.extract('(\d+)').astype(int)
        else:  # One-hot encode other categorical variables
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
    return df



def encode_categorical_DC(df, column):
    """
    Encodes a single categorical column in a DataFrame as ordered (ordinal) or unordered (nominal) 
    based on the column's dtype or unique values.
    
    - For an ordered categorical column, encodes the column with ordinal encoding (integer codes).
    - For an unordered categorical column, applies one-hot encoding.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to be encoded.
    - column (str): The name of the column to encode.

    Returns:
    - pd.DataFrame: The DataFrame with the specified column encoded. 
    """
    # We assume that all dtype object are categorical variables
    if df[column].dtype == 'object':  

        # Some Categorical are ordinal (manually coded)
        if column == 'Age group':
            # Hardcode the order mapping for age categoircal ordered variable
            age_mapping = {
                '≤ 65': 1,
                '66-79': 2,
                '≥ 80': 3
                }
            
            # Apply the map to get the dummy variables
            df.loc[:, 'Age group_Ordinal'] = df['Age group'].map(age_mapping)

        elif column == 'Tumour Stage':
            # Hardcode the order mapping for age categoircal ordered variable
            tumor_mapping = {
                'Stage I': 1,
                'Stage II': 2,
                'Stage III': 3,
                'Stage IV': 4
                }
            
            # Apply the map to get the dummy variables
            df.loc[:, 'Tumour Stage_Ordinal'] = df['Tumour Stage'].map(tumor_mapping)

        else:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
    
    elif df[column].dtype == 'float64':
        # If its a float, we can already do the normal correlation
        print("")

    else:
        print("unexpected type of varibale, check with the encoding process.")


    return df



@measure_time
def calculate_correlations(eigen_genes, trait_dataset, trait_columns):
    """
    Calculates Spearman correlations and p-values between eigengenes and specified trait columns.

    Parameters:
    - eigen_genes (pd.DataFrame): DataFrame of eigengenes with 'Module' as one of the columns.
    - trait_dataset (pd.DataFrame): DataFrame of traits, which may contain categorical variables.
    - trait_columns (list of str): List of column names from 'trait_dataset' to calculate correlations with.

    Returns:
    - Tuple of DataFrames: (correlations, p_values) wtranscriptomics_dataset_filteredhere each DataFrame contains the Spearman correlation
      coefficients and p-values between each module's eigengene and the specified traits.
    """
    correlations = pd.DataFrame()
    p_values = pd.DataFrame()

    for trait in trait_columns:
        # Get encoded traits to be able to calculate Spearman Correlation
        trait_dataset = encode_categorical(trait_dataset, trait)
        trait_data = trait_dataset.filter(like=trait)  

        for module in eigen_genes['Module'].unique():
            # Get the Expression Profile vector for each EigenGene (representing each module). 
            module_data = eigen_genes[eigen_genes['Module'] == module].drop('Module', axis=1).T
            module_data.columns = ['eigengene']

            # Calculate correlations for all samples and all eigengenes
            for trait_name in trait_data.columns:
                cor, p_val = stats.spearmanr(module_data['eigengene'], trait_data[trait_name])
                correlations.loc[module, trait_name] = cor
                p_values.loc[module, trait_name] = p_val

    return correlations, p_values



@measure_time
def eigen_trait_correlations_DC(eigen_genes, trait_dataset, trait_columns):
    """
    Calculates Spearman/Pearson correlations and p-values between eigengenes and specified trait columns.
    Works with different types of encoding

    Parameters:
    - eigen_genes (pd.DataFrame): DataFrame of eigengenes with 'Module' as one of the columns.
    - trait_dataset (pd.DataFrame): DataFrame of traits, which may contain categorical variables.
    - trait_columns (list of str): List of column names from 'trait_dataset' to calculate correlations with.

    Returns:
    - Tuple of DataFrames: (correlations, p_values) wtranscriptomics_dataset_filteredhere each DataFrame contains the Spearman correlation
      coefficients and p-values between each module's eigengene and the specified traits.
    """
    correlations = pd.DataFrame()
    p_values = pd.DataFrame()

    for trait in trait_columns:
        # Get encoded traits to be able to calculate Spearman Correlation
        trait_dataset_encoded = encode_categorical_DC(trait_dataset.copy(), trait)
        trait_data = trait_dataset_encoded.filter(like=trait)  

        for module in eigen_genes['Module'].unique():
            # Get the Expression Profile vector for each EigenGene (representing each module). 
            module_data = eigen_genes[eigen_genes['Module'] == module].drop('Module', axis=1).T
            module_data.columns = ['eigengene']

            for trait_name in trait_data.columns:
                if 'Ordinal' in trait_name or trait_dataset_encoded[trait_name].dtype in ['int64', 'float64']:
                    # Use Pearson correlation for ordinal or continuous variables
                    cor, p_val = stats.pearsonr(module_data['eigengene'], trait_data[trait_name])
                else:
                    # Use Spearman correlation for categorical variables
                    cor, p_val = stats.spearmanr(module_data['eigengene'], trait_data[trait_name])


                correlations.loc[module, trait_name] = cor
                p_values.loc[module, trait_name] = p_val

    return correlations, p_values


################################################################################################





################################################################################################
#                                    SURVIVAL PROBABILITY
################################################################################################

def survival_probability(correlations, corr_th, expression_profiles, survival_dataset, figures_dir):
    """
    Analyzes survival probabilities based on the median expression levels of pacients for the genes in the 
    selected modules.
    Median is used as per the paper 'The proteomic landscape of soft tissue sarcomas'.

    It identifies modules with correlations above a specified threshold, stratifies patients into tertiles based 
    on their median gene expression within these modules, and plots Kaplan-Meier survival curves for each tertile group.

    Parameters:
    - correlations (DataFrame): A DataFrame containing correlations between modules and a clinical trait.
    - corr_th (float): The correlation threshold for selecting modules of interest.
    - expression_profiles (DataFrame): Gene expression profiles with columns for 'Module', 'Gene Name', and expression levels.
    - survival_dataset (DataFrame): Patient survival data, including 'Overall survival days' and 'Vital Status'.
    
    Returns:
    - Generates Kaplan-Meier survival plots for high-correlation modules.
    """
    high_corr_modules = correlations[(correlations > corr_th) | (correlations < -corr_th)].dropna(how='all', axis=0)
    high_corr_modules = high_corr_modules.index.tolist()


    # Iterate over each selected module to make further analysis
    for module in high_corr_modules:
        # Build the full expression profile for each module
        module_expression_profile = expression_profiles[expression_profiles['Module'] == module]
        module_expression_profile = module_expression_profile.drop(columns=['Module'])
        module_expression_profile.set_index('Gene Name', inplace=True)
        module_expression_profile = module_expression_profile.T

        # Calculate the median expression profile of each pacient in this module (cluster of genes)
        pacients_median_module_expression = pd.DataFrame(module_expression_profile.median(axis=1), columns=['MedianExpressionValue']) 

        # Perform subgroup identification by tertile stratification dynamic
        # Calculate the number of unique bins without dropping or assigning labels
        bin_edges = pd.qcut(pacients_median_module_expression['MedianExpressionValue'], 3, duplicates='drop', retbins=True)[1]

        # Generate labels based on the number of bins - 1 (since edges are always one more than the number of bins)
        num_bins = len(bin_edges) - 1
        if num_bins == 1:
            print(f"The module with ID {module} does not have enough Genes to do tertile stratification.")
            continue
        
        labels = ['Low', 'Medium', 'High'][:num_bins]

        # Now apply qcut with the dynamically determined labels
        pacients_median_module_expression['TertileGroup'] = pd.qcut(pacients_median_module_expression['MedianExpressionValue'], 3, labels=labels, duplicates='drop')

        # Merge with the survival days dataset for further analysis
        module_survival_stratification = pd.merge(pacients_median_module_expression, survival_dataset, left_index=True, right_index=True)

        # Identify and remove rows with NaN values in 'Overall survival days' or 'DeathEvent'
        module_survival_stratification = module_survival_stratification.dropna(subset=['Overall survival days', 'Vital Status'])


        ## Kaplan–Meier survival plot of MFS across the three subgroups for each module
        # Dummify Vital Status into DeathEvenet, identifying as 1 pacients that died, and 0 pacients alive.
        module_survival_stratification['Vital Status'] = module_survival_stratification['Vital Status'].map({'Dead': 1, 'Alive': 0})

        # Rename the column 'Vital Status' to 'DeathEvent'
        module_survival_stratification.rename(columns={'Vital Status': 'DeathEvent'}, inplace=True)

        # Initialize the Kaplan-Meier fitter
        kmf = KaplanMeierFitter()

        plt.figure(figsize=(10, 6))

        # Plot survival curves for each group
        for group in labels:
            # Select data for the group
            group_data = module_survival_stratification[module_survival_stratification['TertileGroup'] == group]
            
            # Fit data
            kmf.fit(durations=group_data['Overall survival days'], event_observed=group_data['DeathEvent'], label=group)
            
            # Plot the survival curve
            kmf.plot_survival_function()

        title = 'Survival by Tertile Groups in Module ' + str(module)
        plt.title(title)
        plt.xlabel('Time in Days')
        plt.ylabel('Survival Probability')
        plt.savefig(figures_dir + title, dpi=DPI_GENERAL)
        plt.show()


################################################################################################





################################################################################################
#                               FULL and PARTIAL EXECUTIONS
################################################################################################

def run_full_WGCNA(transcriptomics_dataset, expression_th, trait_dataset, want_plots, figures_dir, \
                   RsquaredCut, MeanCut, block_size_scalefit, adjacency_type, TOMDenom, \
                   height_percentile, min_memb_cluster):
    ### Step 1: Data Preprocessing (Normalization)
    print(f"{BOLD}{OKBLUE}Step 1{ENDC}")
    transcriptomics_dataset_filtered, trait_dataset_filtered = preprocess_TPM_outlier_deletion(transcriptomics_dataset, expression_th, trait_dataset)


    ### Step 2: Constructing a Co-expression Similarity Matrix (Correlation Matrix)
    print(f"{BOLD}{OKBLUE}\n\nStep 2{ENDC}")
    correlation_matrix_np = correlation_matrix(transcriptomics_dataset_filtered, want_plots, figures_dir)
    matrix_np_check(correlation_matrix_np, 1, -1, 1)


    ### Step 3: Transforming into an adjacency matrix using a soft threshold power
    print(f"{BOLD}{OKBLUE}\n\nStep 3{ENDC}")
    optimal_power = pickSoftThreshold(correlation_matrix_np, transcriptomics_dataset_filtered, RsquaredCut, MeanCut, True, figures_dir, block_size_scalefit)

    adjacency_matrix_np = adjacencyM_from_correlationM(correlation_matrix_np, optimal_power, adjacency_type, want_plots, figures_dir)
    matrix_np_check(adjacency_matrix_np, 1, 0, 1)


    ### Step 4: Converting adjacency matrix into a topological overlap matrix (TOM)
    print(f"{BOLD}{OKBLUE}\n\nStep 4{ENDC}")
    simTOM_np = calculate_tom(adjacency_matrix_np, TOMDenom, adjacency_type, want_plots, figures_dir)
    dissTOM_np = 1 - simTOM_np
    matrix_np_check(simTOM_np, 1, 0, 1)


    ### Step 5: Hierarchical clustering
    print(f"{BOLD}{OKBLUE}\n\nStep 5{ENDC}")
    linkage_matrix = hierarchical_clustering(dissTOM_np, want_plots, figures_dir)


    ### Step 6: Module identification
    print(f"{BOLD}{OKBLUE}\n\nStep 6{ENDC}")
    module_assignment, cut_height = identify_modules_simple_version(linkage_matrix, height_percentile, min_memb_cluster)
    module_assignment.insert(0, 'Gene Name', list(transcriptomics_dataset_filtered))


    ### Step 7: Calculate EigenGenes for all identified Modules
    print(f"{BOLD}{OKBLUE}\n\nStep 7{ENDC}")
    expression_profiles = expression_profile_for_cluster(module_assignment, transcriptomics_dataset_filtered)

    eigen_genes = calculate_eigen_genes(expression_profiles, want_plots, figures_dir)


    # Step 8
    print(f"{BOLD}{OKBLUE}\n\nStep 8{ENDC}")
    trait_columns = list(trait_dataset_filtered.columns[1:] )
    correlations, p_values = eigen_trait_correlations_DC(eigen_genes, trait_dataset_filtered, trait_columns)



    print(f"{BOLD}{OKBLUE}Done\n\n{ENDC}")

    return correlations, p_values



def run_partialA_WGCNA(transcriptomics_dataset, expression_th, trait_dataset, want_plots, figures_dir, \
                   RsquaredCut, MeanCut, block_size_scalefit, adjacency_type, TOMDenom):
    ### Step 1: Data Preprocessing (Normalization)
    print(f"{BOLD}{OKBLUE}Step 1{ENDC}")
    transcriptomics_dataset_filtered, trait_dataset_filtered = preprocess_TPM_outlier_deletion(transcriptomics_dataset, expression_th, trait_dataset)


    ### Step 2: Constructing a Co-expression Similarity Matrix (Correlation Matrix)
    print(f"{BOLD}{OKBLUE}\n\nStep 2{ENDC}")
    correlation_matrix_np = correlation_matrix(transcriptomics_dataset_filtered, want_plots, figures_dir)
    matrix_np_check(correlation_matrix_np, 1, -1, 1)


    ### Step 3: Transforming into an adjacency matrix using a soft threshold power
    print(f"{BOLD}{OKBLUE}\n\nStep 3{ENDC}")
    optimal_power = pickSoftThreshold(correlation_matrix_np, transcriptomics_dataset_filtered, RsquaredCut, MeanCut, True, figures_dir, block_size_scalefit)

    adjacency_matrix_np = adjacencyM_from_correlationM(correlation_matrix_np, optimal_power, adjacency_type, want_plots, figures_dir)
    matrix_np_check(adjacency_matrix_np, 1, 0, 1)


    ### Step 4: Converting adjacency matrix into a topological overlap matrix (TOM)
    print(f"{BOLD}{OKBLUE}\n\nStep 4{ENDC}")
    simTOM_np = calculate_tom(adjacency_matrix_np, TOMDenom, adjacency_type, want_plots, figures_dir)
    dissTOM_np = 1 - simTOM_np
    matrix_np_check(simTOM_np, 1, 0, 1)


    ### Step 5: Hierarchical clustering
    print(f"{BOLD}{OKBLUE}\n\nStep 5{ENDC}")
    linkage_matrix = hierarchical_clustering(dissTOM_np, want_plots, figures_dir)


    return linkage_matrix, transcriptomics_dataset_filtered, trait_dataset_filtered



def run_partialB_WGCNA(linkage_matrix, transcriptomics_dataset_filtered, \
                       trait_dataset_filtered, want_plots, figures_dir, \
                        height_percentile, min_memb_cluster):

    ### Step 6: Module identification
    print(f"{BOLD}{OKBLUE}\n\nStep 6{ENDC}")
    module_assignment, cut_height = identify_modules_simple_version(linkage_matrix, height_percentile, min_memb_cluster)
    module_assignment.insert(0, 'Gene Name', list(transcriptomics_dataset_filtered))


    ### Step 7: Calculate EigenGenes for all identified Modules
    print(f"{BOLD}{OKBLUE}\n\nStep 7{ENDC}")
    expression_profiles = expression_profile_for_cluster(module_assignment, transcriptomics_dataset_filtered)

    eigen_genes = calculate_eigen_genes(expression_profiles, want_plots, figures_dir)


    # Step 8
    print(f"{BOLD}{OKBLUE}\n\nStep 8{ENDC}")
    trait_columns = list(trait_dataset_filtered.columns[1:] )
    correlations, p_values = eigen_trait_correlations_DC(eigen_genes, trait_dataset_filtered, trait_columns)



    print(f"{BOLD}{OKBLUE}Done\n\n{ENDC}")

    return module_assignment, correlations, p_values