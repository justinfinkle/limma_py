__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def make_association_dict(association_file, term='TF', association='Gene', sep='\t'):
    """
    Read an association file and turn it into a dictionary
    :param association_file:
    :param term:
    :param association:
    :param sep:
    :return:
    """
    # Read into a dataframe
    association_df = pd.read_csv(association_file, sep=sep)

    term_dict = {term.upper(): df_subset[association].tolist() for term, df_subset in association_df.groupby(term)}

    # todo: this needs to be generalized
    association_dict = {gene.split(';')[0].upper(): df_subset["TF"].tolist() for gene, df_subset in
                        association_df.groupby("Gene")}

    return term_dict, association_dict


def convert_gene_to_tf(gene_list, gene_dict):
    # todo: this needs to be generalized
    """
    Convert a list of genes to a list of transcription factors
    :param gene_list:
    :param gene_dict:
    :return:
    """
    gene_list = [g.upper() for g in gene_list]
    tf_list = []
    count = 0
    for gene in gene_list:
        if gene in gene_dict.keys():
            count += 1
            tf_list = tf_list+gene_dict[gene]
        else:
            continue

    tf_dict = {tf: [] for tf in tf_list}

    # Add genes associated with each tf
    for k in tf_dict.keys():
        for gene in gene_list:
            if gene in gene_dict.keys():
                if k in gene_dict[gene]:
                    tf_dict[k].append(gene)
                else:
                    continue
            else:
                continue

    return tf_list, tf_dict


def calculate_study_enrichment(study_count, study_assoc_dict, bg_count, bg_assoc_dict, fdr=0.05):
    """
    Calculate the enrichment of terms in the study
    :param study_assoc_dict: list
        Terms found in the study
    :param bg_assoc_dict: list
        Terms found in the background
    :param fdr: float
        The expected false discovery rate for which to correct
    :return: dataframe
    """

    # List of unique terms in the study
    study_term_set = list(set(study_assoc_dict.keys()))

    # Calculate p_value for each term
    p_values = np.array([fisher_score(tf, study_assoc_dict, bg_assoc_dict, study_count, bg_count) for tf in study_term_set])

    # Bonferroni correction
    corrected_p = p_values*len(study_term_set)

    # Make results table
    results_table = pd.DataFrame(np.vstack((p_values, corrected_p)).T, columns=['p_uncorrected', 'p_bonferroni'])
    results_table.insert(0, 'TF', study_term_set)
    results_table.sort_values('p_uncorrected', inplace=True)

    #FDR correcton
    results_table['FDR_thresh'] = np.arange(1, len(results_table)+1)/float(len(results_table))*fdr
    results_table['FDR_reject'] = results_table['p_uncorrected'] < results_table['FDR_thresh']
    results_table.sort_values('FDR_reject', ascending=False, inplace=True)
    return results_table


def fisher_score(x, study_dict, bg_dict, study_count, bg_count):
    """
    Get the fisher exact p-value of a term in a list compared to the background
    :param x: str
        The label that is being checked for enrichment in the study.
    :param study_dict: dict
        Terms found in the study
    :param bg_dict: dict
        Terms found in the background
    :return: float
        The p-value associated with the term
    """
    c_table = make_contingency_table(x, study_dict, bg_dict, study_count, bg_count)

    # One sided test looking for over-representation so use 'greater'
    _, pvalue = fisher_exact(c_table, alternative='greater')
    return pvalue


def make_contingency_table(x, study_dict, bg_dict, study_count, bg_count):
    """
    Makes a 2x2 contingency table
    :param x: str
        The label that is being checked for enrichment in the study.
    :param study_dict: list
        Terms found in the study
    :param bg_dict: list
        Terms found in the background
    :return: array
        The 2x2 contingency table used for calculating enrichment
    """
    # Initialize table
    study_set = set(study_dict[x])
    not_selected = bg_count-study_count
    bg_set = set(bg_dict[x]).difference(study_set)

    contingency_table = np.zeros((2, 2))
    selected_with_property = len(study_set)
    selected_without_property = study_count - selected_with_property
    not_selected_with_property = len(bg_set)
    not_selected_without_property = not_selected-not_selected_with_property

    # count_in_background = max(0, background_list.count(x)-match_in_study)
    contingency_table[0, 0] = selected_with_property
    contingency_table[0, 1] = selected_without_property
    contingency_table[1, 0] = not_selected_with_property
    contingency_table[1, 1] = not_selected_without_property
    return contingency_table
