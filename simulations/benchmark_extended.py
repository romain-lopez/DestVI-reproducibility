#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that evaluates model on simulated data.

Created on 2020/02/03
@author romain_lopez
"""

import os
import click
import numpy as np

from utils import get_mean_normal, find_location_index_cell_type, metrics_vector, discrete_histogram
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


param_path = "input_data/"
PCA_path = param_path + "grtruth_PCA_extended.npz"


@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input data directory')
@click.option('--model-subdir', type=click.STRING, default="out/", help='input model subdirectory')
@click.option('--model-string', type=click.STRING, default="description", help='input model description')
@click.option('--missing-ct', type=click.STRING, default="none", help='is something missing')

def main(input_dir, model_subdir, model_string, missing_ct):
    # Directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    # Load data
    grtruth_PCA = np.load(PCA_path)
    mean_, components_ = grtruth_PCA["mean_"], grtruth_PCA["components_"]

    C = components_.shape[0]
    if missing_ct == "sc":
        C -= 1
    # C = 5
    D = components_.shape[1]
    G = components_.shape[2]
    sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    # Create groundtruth
    logger.info("simulate cell-type specific gene expression for abundant cell types in abundant spots (used for imputation)")
    threshold_gt = 0.4
    spot_selection = np.where(st_adata.obsm["cell_type"].max(1) > threshold_gt)[0]
    s_location = st_adata.obsm["locations"][spot_selection]
    s_ct = st_adata.obsm["cell_type"].iloc[spot_selection].values.argmax(1)
    s_gamma = st_adata.obsm["gamma"][spot_selection]
    # get normal means
    s_groundtruth = get_mean_normal(s_ct[:, None], s_gamma[:, None], mean_, components_)[:, 0, :]
    s_groundtruth[s_groundtruth < 0] = 0
    s_groundtruth = np.expm1(s_groundtruth)
    s_groundtruth = s_groundtruth / np.sum(s_groundtruth, axis=1)[:, np.newaxis]

    if "DestVI" in model_string:
        # first load the model
        spatial_model = DestVI.load(input_dir+model_subdir, st_adata)

        # second get the proportion estimates
        proportions = spatial_model.get_proportions().values
        agg_prop_estimates = proportions

        # third impute at required locations
        # for each cell type, query the model at certain locations and compare to groundtruth
        # create a global flush for comparaison across cell types
        imputed_expression = np.zeros_like(s_groundtruth)
        for ct in range(C):
            indices, _ = find_location_index_cell_type(st_adata.obsm["locations"], ct, 
                                                s_location, s_ct)
            expression = spatial_model.get_scale_for_ct(spatial_model.cell_type_mapping[ct], indices=indices).values
            normalized_expression = expression / np.sum(expression, axis=1)[:, np.newaxis]
            # flush to global
            indices_gt = np.where(s_ct == ct)[0]
            imputed_expression[indices_gt] = normalized_expression

    elif "Stereoscope" in model_string:
        # first load the model
        spatial_model = SpatialStereoscope.load(input_dir+model_subdir, st_adata)
        index = int(model_string[-1])
        nb_sub_ct = st_adata.uns["target_list"][index]
        key_clustering = st_adata.uns["key_clustering"][index]

        # second get the proportion estimates
        proportions = spatial_model.get_proportions().values
        agg_prop_estimates = proportions[:, ::nb_sub_ct]
        for i in range(1, nb_sub_ct):
            agg_prop_estimates += proportions[:, i::nb_sub_ct]

        # third impute at required locations
        # for each cell type, query the model at certain locations and compare to groundtruth
        # create a global flush for comparaison across cell types
        imputed_expression = np.zeros_like(s_groundtruth)
        for ct in range(C):
            indices, _ = find_location_index_cell_type(st_adata.obsm["locations"], ct, 
                                                s_location, s_ct)
            n_indices = indices.shape[0]
            if nb_sub_ct > 1:
                # hierarchical clusters in Stereoscope
                partial_cell_type = proportions[indices, nb_sub_ct*ct:nb_sub_ct*ct+nb_sub_ct] 
                partial_cell_type /= np.sum(partial_cell_type, axis=1)[:, np.newaxis] # shape (cells, nb_sub_ct)
                expression = np.zeros(shape=(n_indices, G))
                for t in range(nb_sub_ct):
                    mask = sc_adata.obs[key_clustering] == t + nb_sub_ct * ct
                    average = np.mean(sc_adata.X[mask].A, axis=0)
                    expression += partial_cell_type[:, [t]] * average
            else:
                # smooth sc-gene expression in cell type
                mask = sc_adata.obs["cell_type"] == ct
                expression = np.mean(sc_adata.X[mask].A, axis=0)
                expression = np.repeat(expression[np.newaxis, :], n_indices, axis=0)

            # old imputation code
            # for t in range(nb_sub_ct):
            #     y = np.array(indices.shape[0] * [spatial_model.cell_type_mapping[t + nb_sub_ct * ct]])
            #     expression += partial_cell_type[:, [t]] * spatial_model.get_scale_for_ct(y)

            normalized_expression = expression / np.sum(expression, axis=1)[:, np.newaxis]
            # flush to global
            indices_gt = np.where(s_ct == ct)[0]
            imputed_expression[indices_gt] = normalized_expression


            # normalized_expression = expression / np.sum(expression, axis=1)[:, np.newaxis]
            # flush to global
            indices_gt = np.where(s_ct == ct)[0]
            imputed_expression[indices_gt] = normalized_expression

    elif model_string in ["Harmony", "Scanorama", "scVI"]:
        # in this model, we must calculate everything via nearest neighbors
        # second get the proportion estimates and get scores
        k_proportions = 50
        embed_ = np.load(input_dir+model_subdir + '/embedding.npz')
        embedding_sc = embed_["embedding_sc"]
        embedding_st = embed_["embedding_st"]
        tree = KDTree(embedding_sc)
        neighbors = tree.query(embedding_st, k=k_proportions, return_distance=False)
        ct_counts = sc_adata.obs["cell_type"][neighbors.reshape((-1))].values.reshape((-1, k_proportions))
        agg_prop_estimates = discrete_histogram(ct_counts, C)
        # third impute at required locations
        k_expression= 50
        all_res = []
        # for each cell type, query the model at certain locations and compare to groundtruth
        # create a global flush for comparaison across cell types
        imputed_expression = np.zeros_like(s_groundtruth)
        for ct in range(C):
            # get indices of interest (=place to impute) for the given cell type
            indices, _ = find_location_index_cell_type(st_adata.obsm["locations"], ct, 
                                                s_location, s_ct)
            n_indices = indices.shape[0]
            # build a KDTree containing only this cell type in the single-cell data
            mask = sc_adata.obs["cell_type"] == ct
            sliced_adata = sc_adata[mask].copy()
            tree = KDTree(embedding_sc[mask])
            neighbors = tree.query(embedding_st[indices], k=k_expression, return_distance=False)          
            expression = sliced_adata.X[neighbors.reshape((k_expression * n_indices,))].A.reshape((n_indices, k_expression, -1))
            expression = expression.mean(1)
            # import pdb; pdb.set_trace()
            normalized_expression = expression / np.sum(expression, axis=1)[:, np.newaxis]
            # flush to global
            indices_gt = np.where(s_ct == ct)[0]
            imputed_expression[indices_gt] = normalized_expression

    elif "RCTD" in model_string or "Spotlight" in model_string or "Seurat" in model_string or "cell2location" in model_string:
        index = int(model_string[-1])
        nb_sub_ct = st_adata.uns["target_list"][index]
        key_clustering = st_adata.uns["key_clustering"][index]

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
            proportions = reorder_df_columns(proportions).values
        
        agg_prop_estimates = proportions[:, ::nb_sub_ct]
        for i in range(1, nb_sub_ct):
            agg_prop_estimates += proportions[:, i::nb_sub_ct]
        # impute all cell types
        imputed_expression = np.zeros_like(s_groundtruth)
        for ct in range(C):
            # get indices of interest (=place to impute) for the given cell type
            indices, _ = find_location_index_cell_type(st_adata.obsm["locations"], ct, 
                                                s_location, s_ct)
            n_indices = indices.shape[0]
            if nb_sub_ct > 1:
                # hierarchical clusters
                partial_cell_type = proportions[indices, nb_sub_ct*ct:nb_sub_ct*ct+nb_sub_ct]
                total_partial = np.sum(partial_cell_type, axis=1)
                # sometimes it is not affected to any cell type -> just mix them
                partial_cell_type[total_partial == 0] = 1
                total_partial[total_partial == 0] = C
                partial_cell_type /= total_partial[:, np.newaxis] # shape (cells, nb_sub_ct)
                # impute for all sub-cell types
                expression = np.zeros(shape=(n_indices, imputed_expression.shape[1]))
                for t in range(nb_sub_ct):
                    mask = sc_adata.obs[key_clustering] == t + nb_sub_ct * ct
                    average = np.mean(sc_adata.X[mask].A, axis=0)
                    expression += partial_cell_type[:, [t]] * average

            else:
                # smooth sc-gene expression in cell type
                mask = sc_adata.obs["cell_type"] == ct
                expression = np.mean(sc_adata.X[mask].A, axis=0)
                expression = np.repeat(expression[np.newaxis, :], n_indices, axis=0)

            normalized_expression = expression / np.sum(expression, axis=1)[:, np.newaxis]
            # flush to global
            indices_gt = np.where(s_ct == ct)[0]
            imputed_expression[indices_gt] = normalized_expression
        
    else:
        raise ValueError("unknown model string")

    # score these predictions against GT
    all_res = []
    all_res_long = []
    for ct in range(C):
        # get local scores
        indices_gt = np.where(s_ct == ct)[0]
        # potentially filter genes for local scores only
        gene_list = np.unique(np.hstack([np.where(components_[ct, i] != 0)[0] for i in range(D)]))
        res = metrics_vector(s_groundtruth[indices_gt], imputed_expression[indices_gt], scaling=2e5, feature_shortlist=gene_list)
        res_long = metrics_vector(s_groundtruth[indices_gt], imputed_expression[indices_gt], scaling=2e5)
        all_res.append(pd.Series(res))
        all_res_long.append(pd.Series(res_long))
    
    all_res.append(pd.Series(metrics_vector(s_groundtruth, imputed_expression, scaling=2e5)))
    all_res = all_res + all_res_long
    df = pd.concat(all_res, axis=1)
    if st_adata.obsm["cell_type"].shape[1] == agg_prop_estimates.shape[1]:
        # same cell types
        prop_score = metrics_vector(st_adata.obsm["cell_type"].values, agg_prop_estimates)
    elif st_adata.obsm["cell_type"].shape[1] == agg_prop_estimates.shape[1] + 1:
        # missing ct in single-cell
        prop_score = metrics_vector(st_adata.obsm["cell_type"].values[:, :-1], agg_prop_estimates)
    elif st_adata.obsm["cell_type"].shape[1]+1 == agg_prop_estimates.shape[1]:
        # missing ct in spatial
        prop_score = metrics_vector(st_adata.obsm["cell_type"].values, agg_prop_estimates[:, :-1])
    else:
        raise ValueError("wrong shapes")
    df = pd.concat([df, pd.Series(prop_score)], axis=1)
    df.columns = ["ct" + str(i) for i in range(C)] + ["ct_long" + str(i) for i in range(C)] +["allct", "proportions"]    
    df.to_csv(input_dir+model_subdir + "/result.csv")


if __name__ == '__main__':
    main()