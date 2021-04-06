#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that simulates data for benchmarking of DestVI.

Created on 2020/02/03
@author romain_lopez
"""

import os
import click
import numpy as np
np.random.seed(0)
from logzero import logger

from utils import get_mean_normal, categorical
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import anndata
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
torch.manual_seed(0)
from torch.distributions import Gamma

param_path = "input_data/"
PCA_path = param_path + "grtruth_PCA.npz"

@click.command()
@click.option('--output-dir', type=click.STRING, default="out/", help='output directory')
@click.option("--input-file", type=click.STRING, default=PCA_path, help="input npz file defining cell types")
@click.option('--lam-ct', type=click.FLOAT, default=0.1, help='Bandwitdh tweak for cell type proportion')
@click.option('--temp-ct', type=click.FLOAT, default=1, help='Temperature tweak for cell type proportion')
@click.option('--lam-gam', type=click.FLOAT, default=0.5, help='Bandwitdh tweak for gamma values')
@click.option('--sf-gam', type=click.FLOAT, default=15, help='Variance scaling tweak for gamma values')
@click.option('--threshold-gt', type=click.FloatRange(min=0, max=1), default=0.4, help='Threshold proportion for defining GE groundtruth')
def main(output_dir, input_file, lam_ct, temp_ct, lam_gam, sf_gam, threshold_gt):
    # parameters 
    K = 100
    K_sampled = 20
    grid_size = 10
    # Directory management
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.isdir(output_dir):
        logger.info("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    else:
        logger.info(F"Found directory at:{output_dir}")

    grtruth_PCA = np.load(input_file)
    mean_, components_ = grtruth_PCA["mean_"], grtruth_PCA["components_"]

    inv_dispersion = np.load(param_path + "inv-dispersion.npy")

    C = components_.shape[0]
    D = components_.shape[1]
    logger.info("get spatial patterns")
    locations, freq_sample, gamma = generate_spatial_information(C=C, D=D, grid_size=grid_size, 
        lam_ct=lam_ct, temp_ct=temp_ct, lam_gam=lam_gam, sf_gam=sf_gam, savefig=output_dir)

    logger.info("generate single-cell data on the spatial grid, relating the spatial patterns to the sPCA model")
    cell_types_sc = categorical(freq_sample, K)
    gamma_sc = gamma[:, None, :].repeat(K, axis=1)
    location_sc = locations[:, None, :].repeat(K, axis=1)
    # get means of the Gaussian using the sPCA model
    mean_normal = get_mean_normal(cell_types_sc, gamma_sc, mean_, components_)
    # convert back to count distribution and sample from Poisson
    mean_normal[mean_normal <= 0] = np.min(mean_normal[mean_normal > 0]) * 0.01
    transformed_mean = np.expm1(mean_normal)

    # dispersion was learned on the single-cell data. 
    # this simulation might have different library sizes
    # we must match them so that the dispersion estimates make sense (i.e., genes are as overpoissonian as in the experiments)
    inv_dispersion *= 1e2

    if True:
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        gamma_s = Gamma(concentration=torch.tensor(inv_dispersion), 
            rate=torch.tensor(inv_dispersion) / torch.tensor(transformed_mean)).sample()
        mean_poisson = torch.clamp(gamma_s, max=1e8).cpu().numpy()
        transformed_mean = mean_poisson

    samples = np.random.poisson(lam=transformed_mean)

    logger.info("dump scRNA-seq")
    sc_anndata = anndata.AnnData(X=csr_matrix(samples[:, :K_sampled].reshape((-1, samples.shape[-1]))))
    sc_anndata.obs["cell_type"] = cell_types_sc[:, :K_sampled].reshape(-1, 1)
    sc_anndata.obs["n_counts"] = np.sum(sc_anndata.X.A, axis=1)
    sc_anndata.obsm["gamma"] = gamma_sc[:, :K_sampled].reshape(-1, gamma.shape[-1])
    sc_anndata.obsm["locations"] = location_sc[:, :K_sampled].reshape(-1, 2)

    logger.info("cluster cell-type-specific single-cell data (used in discrete deconvolution baselines)")
    # cluster the single-cell data using sklearn
    target_list = [2, 4, 8, 16]
    key_list = ["cell_type"]
    hier_labels_sc = np.zeros((sc_anndata.n_obs, len(target_list)))
    for ct in range(5):
        slice_ind = np.where(sc_anndata.obs["cell_type"] == ct)
        slice_counts = sc_anndata.X[slice_ind].A
        slice_normalized = slice_counts / np.sum(slice_counts, axis=1)[:, np.newaxis]
        slice_embedding = PCA(n_components=10).fit_transform(np.log(1 + 1e4 * slice_normalized))
        knn_graph = kneighbors_graph(slice_embedding, 30, include_self=False)
        for i, target in enumerate(target_list):
            labels = AgglomerativeClustering(n_clusters=target, connectivity=knn_graph).fit_predict(slice_embedding)
            hier_labels_sc[slice_ind, i] = labels
            
    # aggregate hierarchical labels and append to anndata
    for i, target in enumerate(target_list):
        base_cell_type = sc_anndata.obs["cell_type"]
        sub_cell_type = hier_labels_sc[:, i]
        nb_sub_ct = len(np.unique(sub_cell_type))
        all_cell_type = np.array([base_cell_type[j] * nb_sub_ct + sub_cell_type[j] for j in range(sc_anndata.n_obs)])
        key = str(target) + "th_sub-cell_type"
        sc_anndata.obs[key] = all_cell_type.astype(np.int)
        key_list.append(key)
    # dump keys as well
    sc_anndata.uns["key_clustering"] = key_list
    sc_anndata.uns["target_list"] = [1] + target_list
    sc_anndata.write(output_dir + "sc_simu.h5ad", compression="gzip")

    transformed_mean_st = transformed_mean.mean(1)
    if True:
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        gamma_st = Gamma(concentration=torch.tensor(inv_dispersion), 
            rate=torch.tensor(inv_dispersion) / torch.tensor(transformed_mean_st)).sample()
        mean_poisson_st = torch.clamp(gamma_st, max=1e8).cpu().numpy()
        transformed_mean_st = mean_poisson_st
    
    logger.info("dump spatial")
    samples_st = np.random.poisson(lam=transformed_mean_st)
    st_anndata = anndata.AnnData(X=csr_matrix(samples_st))
    st_anndata.obsm["cell_type"] = freq_sample
    st_anndata.obsm["gamma"] = gamma
    st_anndata.obsm["locations"] = locations
    st_anndata.obs["n_counts"] = np.sum(st_anndata.X, axis=1)
    st_anndata.uns["key_clustering"] = key_list
    st_anndata.uns["target_list"] = [1] + target_list
    st_anndata.write(output_dir + "st_simu.h5ad", compression="gzip")
   

def generate_spatial_information(grid_size, C, lam_ct, temp_ct, lam_gam, D, sf_gam, savefig="out/"):
    locations = np.mgrid[-grid_size:grid_size:0.5, -grid_size:grid_size:0.5].reshape(2,-1).T
    # get the kernel bandwidth for GP simulation
    dist_table = pdist(locations)
    bandwidth = np.median(dist_table)
    # sample from the multivariate GP for cell type
    K = np.exp(- squareform(dist_table)**2 / (lam_ct*bandwidth**2))
    N = K.shape[0]
    sample = np.random.multivariate_normal(np.zeros(N), K, size=C).T
    # get through softmax
    e_sample = np.exp(sample / temp_ct)
    freq_sample = e_sample / np.sum(e_sample, 1)[:, np.newaxis]
    
    # form the multivariate GP covariance for gamma
    K = sf_gam * np.exp(- squareform(dist_table)**2 / (lam_gam*bandwidth**2))
    # get latent variable for each cell types
    gamma = np.random.multivariate_normal(np.zeros(N), K, size=(D)).T

    # plot cell types
    plt.figure(figsize=(10,6))
    for i in range(0, C):
        plt.subplot(231 + i)
        plt.scatter(locations[:, 0], locations[:, 1], c=freq_sample[:, i])
        plt.title("cell type "+str(i))
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(savefig+"cell_type_proportion.png")
    plt.clf()
    # plot gammas
    plt.figure(figsize=(5,5))
    for i in range(D):
        plt.subplot(221 + i)
        plt.scatter(locations[:, 0], locations[:, 1], c=gamma[:, i])
        plt.title("gamma "+str(i))
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(savefig+"gamma.png")
    plt.clf()
    return locations, freq_sample, gamma


if __name__ == '__main__':
    main()