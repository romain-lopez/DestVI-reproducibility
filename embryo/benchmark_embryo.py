#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that evaluates model based on single-cell spatial data.

Created on 2021/07/16
@author romain_lopez
"""

import os
import click
import numpy as np

from utils import metrics_vector, discrete_histogram
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata

import scvi
from scvi.model import DestVI
from scvi.external import SpatialStereoscope

from sklearn.neighbors import KDTree

scvi.settings.reset_logging_handler()
import logging
logger = logging.getLogger("scvi")


@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input data directory')
@click.option('--model-subdir', type=click.STRING, default="out/", help='input model subdirectory')
@click.option('--model-string', type=click.STRING, default="description", help='input model description')
def main(input_dir, model_subdir, model_string):
    # Directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    # Load data
    sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    # Create groundtruth
    logger.info("simulate cell-type specific gene expression for abundant cell types in abundant spots (used for imputation)")
    threshold_gt = 0.2
    spot_selection = {}
    n_spot_select = {}
    s_groundtruth = {}
    ct_list = [("Neuron", 0), ("Radial glia", 2)]
    for ct_name, _ in ct_list:
        spot_selection[ct_name] = np.where(st_adata.obsm["proportions"][ct_name] > threshold_gt)[0]
        n_spot_select[ct_name] = spot_selection[ct_name].shape[0]
        groundtruth = st_adata.obsm[ct_name].iloc[spot_selection[ct_name]].values
        groundtruth = groundtruth / np.sum(groundtruth, axis=1)[:, np.newaxis]
        s_groundtruth[ct_name] = np.nan_to_num(groundtruth)
    interest_genes = st_adata.uns["interest_genes"]
    gene_lookup = [np.where(st_adata.var.index.values == gene)[0][0] for gene in interest_genes]
    # for neurons (0)
    C = st_adata.obsm["proportions"].values.shape[1]

    if "DestVI" in model_string:
        # first load the model
        spatial_model = DestVI.load(input_dir+model_subdir, st_adata)

        # second get the proportion estimates
        proportions = spatial_model.get_proportions().values
        agg_prop_estimates = proportions

        # third impute at required locations
        # for neurons (0), query the model at certain locations and compare to groundtruth
        imputed_expression = {}
        for ct_name, ct in ct_list:
            expression_ = spatial_model.get_scale_for_ct(ct, indices=spot_selection[ct_name]).values
            imputed_expression[ct_name] = expression_ / np.sum(expression_, axis=1)[:, np.newaxis]

    elif "Stereoscope" in model_string:
        # first load the model
        spatial_model = SpatialStereoscope.load(input_dir+model_subdir, st_adata)
        index = int(model_string[-1])
        nb_sub_ct = sc_adata.uns["target_list"][index]

        # second get the proportion estimates
        proportions = spatial_model.get_proportions().values
        agg_prop_estimates = proportions[:, ::nb_sub_ct]
        for i in range(1, nb_sub_ct):
            agg_prop_estimates += proportions[:, i::nb_sub_ct]

        # third impute at required locations
        # for each cell type, query the model at certain locations and compare to groundtruth
            # hierarchical clusters in Stereoscope
        imputed_expression = {}
        for ct_name, ct in ct_list:
            partial_cell_type = proportions[spot_selection[ct_name], nb_sub_ct*ct:nb_sub_ct*ct+nb_sub_ct] 
            partial_cell_type /= np.sum(partial_cell_type, axis=1)[:, np.newaxis] # shape (cells, nb_sub_ct)
            expression = np.zeros(shape=(n_spot_select[ct_name], st_adata.n_vars))
            for t in range(nb_sub_ct):
                y = np.array(n_spot_select[ct_name] * [spatial_model.cell_type_mapping[t + nb_sub_ct * ct]])
                expression += partial_cell_type[:, [t]] * spatial_model.get_scale_for_ct(y)
            imputed_expression[ct_name] = expression / np.sum(expression, axis=1)[:, np.newaxis]
        
    elif model_string in ["Harmony", "Scanorama", "scVI"]:
        # in this model, we must calculate everything via nearest neighbors
        # second get the proportion estimates and get scores
        k_proportions = 10
        embed_ = np.load(input_dir+model_subdir + '/embedding.npz')
        embedding_sc = embed_["embedding_sc"]
        embedding_st = embed_["embedding_st"]
        tree = KDTree(embedding_sc)
        neighbors = tree.query(embedding_st, k=k_proportions, return_distance=False)
        # ct_counts = sc_adata.obs["cell_type"][neighbors.reshape((-1))].values.reshape((-1, k_proportions))
        # agg_prop_estimates = discrete_histogram(ct_counts, C)
        index = 2
        nb_sub_ct = sc_adata.uns["target_list"][index]
        key =  sc_adata.uns["key_clustering"][index]
        ct_counts = sc_adata.obs[key][neighbors.reshape((-1))].values.reshape((-1, k_proportions))
        proportions = discrete_histogram(ct_counts, nb_sub_ct*C)
        agg_prop_estimates = proportions[:, ::nb_sub_ct]
        for i in range(1, nb_sub_ct):
            agg_prop_estimates += proportions[:, i::nb_sub_ct]

        # hierarchical clusters
        imputed_expression = {}
        for ct_name, ct in ct_list:
            partial_cell_type = proportions[spot_selection[ct_name], nb_sub_ct*ct:nb_sub_ct*ct+nb_sub_ct]
            total_partial = np.sum(partial_cell_type, axis=1)
            # sometimes it is not affected to any cell type -> just mix them
            partial_cell_type[total_partial == 0] = 1
            total_partial[total_partial == 0] = C
            partial_cell_type /= total_partial[:, np.newaxis] # shape (cells, nb_sub_ct)
            # impute for all sub-cell types
            expression = np.zeros(shape=(n_spot_select[ct_name], st_adata.n_vars))
            for t in range(nb_sub_ct):
                mask = sc_adata.obs[key] == t + nb_sub_ct * ct
                average = np.mean(sc_adata.X[mask].A, axis=0)
                expression += partial_cell_type[:, [t]] * average
            imputed_expression[ct_name] = expression / np.sum(expression, axis=1)[:, np.newaxis]

        # # third impute at required locations
        # k_expression= 300
        # # for each cell type, query the model at certain locations and compare to groundtruth
        # # build a KDTree containing only this cell type in the single-cell data
        # imputed_expression = {}
        # for ct_name, ct in ct_list:
        #     mask = sc_adata.obs["cell_type"] == ct
        #     sliced_adata = sc_adata[mask].copy()
        #     tree = KDTree(embedding_sc[mask])
        #     neighbors = tree.query(embedding_st[spot_selection[ct_name]], k=k_expression, return_distance=False)          
        #     expression = sliced_adata.X[neighbors.reshape((k_expression * n_spot_select[ct_name],))].A
        #     expression = expression.reshape((n_spot_select[ct_name], k_expression, -1))
        #     expression = expression.mean(1)
        #     imputed_expression[ct_name] = expression / np.sum(expression, axis=1)[:, np.newaxis]


    elif "RCTD" in model_string or "Spotlight" in model_string or "Seurat" in model_string or "cell2location" in model_string:
        index = int(model_string[-1])
        nb_sub_ct = sc_adata.uns["target_list"][index]
        key_clustering = sc_adata.uns["key_clustering"][index]

        # read results from csv file
        # second get the proportion estimates
        if "cell2location" in model_string:
            proportions = pd.read_csv(input_dir + model_subdir + "/results/W_cell_density.csv", index_col=0).values
            proportions = proportions / np.sum(proportions, axis=1)[:, None]
        else:
            import re
            def extract_last_text(text):
                return list(map(int, re.findall(r'\d+', text)))[-1]
            def reorder_df_columns(df):
                df.columns = [extract_last_text(x) for x in df.columns]
                return df.loc[:, range(len(df.columns))]

            proportions = pd.read_csv(input_dir+model_subdir + '/output_weights.csv', index_col=0)
            proportions = reorder_df_columns(proportions)
        # some minor subsampling done in RCTD (removes 1, 2 or 3 cells)
        # import pdb; pdb.set_trace()
        prop_values = pd.DataFrame(data=1/ (C * nb_sub_ct) * np.ones(shape=(st_adata.n_obs, C * nb_sub_ct)), index=st_adata.obs.index)
        # if "Spotlight" in model_string:
        indices = [str(x) for x in proportions.index]
        prop_values.loc[indices] = proportions.values
        proportions = prop_values.values

        agg_prop_estimates = proportions[:, ::nb_sub_ct]
        for i in range(1, nb_sub_ct):
            agg_prop_estimates += proportions[:, i::nb_sub_ct]

        if nb_sub_ct > 1:
            # hierarchical clusters
            imputed_expression = {}
            for ct_name, ct in ct_list:
                partial_cell_type = proportions[spot_selection[ct_name], nb_sub_ct*ct:nb_sub_ct*ct+nb_sub_ct]
                total_partial = np.sum(partial_cell_type, axis=1)
                # sometimes it is not affected to any cell type -> just mix them
                partial_cell_type[total_partial == 0] = 1
                total_partial[total_partial == 0] = C
                partial_cell_type /= total_partial[:, np.newaxis] # shape (cells, nb_sub_ct)
                # impute for all sub-cell types
                expression = np.zeros(shape=(n_spot_select[ct_name], st_adata.n_vars))
                for t in range(nb_sub_ct):
                    mask = sc_adata.obs[key_clustering] == t + nb_sub_ct * ct
                    average = np.mean(sc_adata.X[mask].A, axis=0)
                    expression += partial_cell_type[:, [t]] * average
                imputed_expression[ct_name] = expression / np.sum(expression, axis=1)[:, np.newaxis]


        else:
            imputed_expression = {}
            for ct_name, ct in ct_list:
                # smooth sc-gene expression in cell type
                mask = sc_adata.obs["cell_type"] == ct
                expression = np.mean(sc_adata.X[mask].A, axis=0)
                imputed_expression_ = expression[np.newaxis:, ] / np.sum(expression)
                imputed_expression[ct_name] = np.repeat(imputed_expression_[np.newaxis, :], n_spot_select[ct_name], axis=0)


        
    else:
        raise ValueError("unknown model string")
    # score these predictions against GT
    filt_expression = {}
    for ct_name, ct in ct_list:
        filt_expression[ct_name] = imputed_expression[ct_name][:, gene_lookup]
    corr_neuron_res = pd.Series(metrics_vector(s_groundtruth["Neuron"], filt_expression["Neuron"], scaling=2e5))
    corr_rglia_res = pd.Series(metrics_vector(s_groundtruth["Radial glia"], filt_expression["Radial glia"], scaling=2e5))
    prop_res = pd.Series(metrics_vector(st_adata.obsm["proportions"].values, agg_prop_estimates))

    mask = np.sum(st_adata.obsm["proportions"].values > 0, axis=1) > 1
    f1prop_res = pd.Series(metrics_vector(st_adata.obsm["proportions"].values[mask], agg_prop_estimates[mask]))
    f2prop_res = pd.Series(metrics_vector(st_adata.obsm["proportions"].values[~mask], agg_prop_estimates[~mask]))
    # mask = mask[spot_selection["Neuron"]]
    # f1corr_res = pd.Series(metrics_vector(s_groundtruth[mask], filt_expression[mask], scaling=2e5))
    # f2corr_res = pd.Series(metrics_vector(s_groundtruth[~mask], filt_expression[~mask], scaling=2e5))
    # df = pd.concat([corr_neuron_res, corr_rglia_res, prop_res, f1prop_res, f2prop_res, f1corr_res, f2corr_res], axis=1)
    df = pd.concat([corr_neuron_res, corr_rglia_res, prop_res], axis=1)
    df.columns = ["Neurons", "Radial glia", "proportions"] 
    df.to_csv(input_dir+model_subdir + "/result.csv")


if __name__ == '__main__':
    main()