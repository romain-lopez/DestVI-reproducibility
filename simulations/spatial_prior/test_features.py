# ## Importing packages and data


#!/usr/bin/env python3
# -*- coding: utf-8

import copy
import os
import sys

import anndata
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
import scvi
import seaborn as sns
import torch
from scvi.model import CondSCVI, DestVI, SCVI
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from destvi_spatial import DestVISpatial, CustomTrainingPlan

scvi.settings.reset_logging_handler()
import logging

sys.path.append("/data/yosef2/users/pierreboyeau/DestVI-reproducibility/simulations")
from utils import (
    discrete_histogram,
    find_location_index_cell_type,
    get_mean_normal,
    metrics_vector,
)


# def test_models():
def construct_neighboors(adata, n_neighbors=5):
    locs = adata.obsm["locations"]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(locs)
    idx_to_neighs = nbrs.kneighbors(locs)[1][:, 1:]
    n_indices_ = torch.tensor(idx_to_neighs)
    X = torch.tensor(st_adata.X.todense())
    X_neigh = X[n_indices_]
    return X_neigh.numpy(), n_indices_.numpy()


def construct_spatial_partition(adata, n_cv=5):
    locs = adata.obsm["locations"]
    clust = KMeans(n_clusters=n_cv, n_init=100)
    attribs = clust.fit_predict(locs)
    return attribs


WORKING_DIR = "/data/yosef2/users/pierreboyeau/scvi-tools/simulations_code"
input_dir = os.path.join(WORKING_DIR, "out/")
sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

scvi.data.setup_anndata(sc_adata, labels_key="cell_type")
mapping = sc_adata.uns["_scvi"]["categorical_mappings"]["_scvi_labels"]["mapping"]
# train sc-model
sc_model = CondSCVI(sc_adata, n_latent=4, n_layers=2, n_hidden=128)
sc_model.train(
    max_epochs=1,
    plan_kwargs={"n_epochs_kl_warmup": 2},
    progress_bar_refresh_rate=1,
)

# mean_vprior, var_vprior = sc_model.get_vamp_prior(sc_adata, p=100)
x_n, ind_n = construct_neighboors(st_adata, n_neighbors=5)
attribs = construct_spatial_partition(st_adata)
st_adata.obsm["x_n"] = x_n
st_adata.obsm["ind_n"] = ind_n
scvi.data.setup_anndata(st_adata)

mult_ = 1
amortization = "latent"

gt_props = pd.DataFrame(st_adata.obsm["cell_type"])
gt_props.columns = ["ct0", "ct1", "ct2", "ct3", "ct4"]
gt_props = (
    gt_props.stack()
    .to_frame("proportion_gt")
    .reset_index()
    .rename(columns={"level_0": "spot", "level_1": "celltype"})
)

param_path = "/data/yosef2/users/pierreboyeau/data/spatial_data/"
PCA_path = param_path + "grtruth_PCA.npz"
grtruth_PCA = np.load(PCA_path)
mean_, components_ = grtruth_PCA["mean_"], grtruth_PCA["components_"]

C = components_.shape[0]
D = components_.shape[1]

threshold_gt = 0.4
spot_selection = np.where(st_adata.obsm["cell_type"].max(1) > threshold_gt)[0]
s_location = st_adata.obsm["locations"][spot_selection]
s_ct = st_adata.obsm["cell_type"][spot_selection, :].argmax(1)
s_gamma = st_adata.obsm["gamma"][spot_selection]
# get normal means
s_groundtruth = get_mean_normal(s_ct[:, None], s_gamma[:, None], mean_, components_)[
    :, 0, :
]
s_groundtruth[s_groundtruth < 0] = 0
s_groundtruth = np.expm1(s_groundtruth)
s_groundtruth = s_groundtruth / np.sum(s_groundtruth, axis=1)[:, np.newaxis]


spatial_model_prior = DestVISpatial.from_rna_model(
    st_adata,
    sc_model,
    vamp_prior_p=100,
    amortization=amortization,
    spatial_prior=True, 
    spatial_agg="pair",
    lamb=2.,
)
spatial_model_prior.train(
    max_epochs=2000,
    train_size=1,
    lr=1e-2, 
    n_epochs_kl_warmup=400,
    progress_bar_refresh_rate=0,
)
# df = destvi_get_metrics(spatial_model_prior)

spatial_model_gt = DestVISpatial.from_rna_model(
    st_adata,
    sc_model,
    vamp_prior_p=100,
    amortization=amortization,
    spatial_prior=False,
)
spatial_model_gt.train(
    max_epochs=1,
    train_size=1,
    lr=1e-1,
    n_epochs_kl_warmup=100,
    progress_bar_refresh_rate=1,
)


expression = sc_model.get_normalized_expression()

cts = sc_model.adata.obs["cell_type"]

scvi_model = SCVI(sc_adata, n_latent=2, n_layers=2, n_hidden=128)
scvi_model.train(
    max_epochs=2,
    plan_kwargs={"n_epochs_kl_warmup": 2},
    progress_bar_refresh_rate=1,
)

latent = scvi_model.get_latent_representation()

latent_ = pd.DataFrame(latent)
latent_.index = ["cell" + str(col) for col in latent_.index]
expression_ = expression.T
expression_.columns = ["cell" + str(col) for col in expression_.columns]

import hotspot

hs = hotspot.Hotspot(expression_, model="none", latent=latent_)

hs.create_knn_graph(weighted_graph=False, n_neighbors=3)
hs_results = hs.compute_autocorrelations()

# hs_genes = hs_results.loc[hs_results.FDR < 0.05].index # Select genes
# hs_genes = hs_results.index
# local_correlations = hs.compute_local_correlations(
#     hs_genes, jobs=40
# )  # jobs for parallelization

# modules = hs.create_modules(min_gene_threshold=30, core_only=False, fdr_threshold=0.05)


# gene_train_indices = (
#     modules.groupby(modules)
#     .apply(lambda x: x.sample(frac=0.5).index.to_series().astype(int))
#     .to_frame("indices")
#     .reset_index()
#     .indices.values
# )

nfolds = 2
ngenes = st_adata.X.shape[-1]
heldout_folds = np.arange(nfolds)
# gene_folds = np.isin(np.arange(ngenes), gene_train_indices)
gene_folds = np.isin(np.arange(ngenes), np.random.permutation(ngenes)[:200])


for heldout in heldout_folds[:-1]:
    training_mask = gene_folds != heldout
    training_mask = torch.tensor(training_mask)
    test_mask = ~training_mask
    print(training_mask.sum(), test_mask.sum())


# # # ### Grid search
# # training_mask = gene_folds != heldout
# # training_mask = torch.tensor(training_mask)
# # test_mask = ~training_mask
lamb = 1.0
spatial_model = DestVISpatial.from_rna_model(
    st_adata,
    sc_model,
    vamp_prior_p=100,
    amortization=amortization,
    spatial_prior=True,
    spatial_agg="pair",
    lamb=lamb,
    training_mask=training_mask,
)

# Step 1: training genes
spatial_model.train(
    max_epochs=1,
    train_size=1,
    lr=1e-2,
    n_epochs_kl_warmup=400,
    plan_kwargs=dict(
        loss_mask=training_mask,
    ),
    progress_bar_refresh_rate=1,
    # plan_class=CustomTrainingPlan,
)
rec_loss, rec_loss_all = spatial_model.get_metric()


# # pass_results = pd.DataFrame(spatial_model.history).assign(
# #     heldout=heldout,
# #     lamb=lamb,
# #     train_phase=True,
# # )
# # cv_results = cv_results.append(pass_results, ignore_index=True)

# # # Step 2: heldout genes
myparameters = [spatial_model.module.eta] + [spatial_model.module.beta]
myparameters = filter(lambda p: p.requires_grad, myparameters)
spatial_model.train(
    max_epochs=2,
    train_size=1,
    progress_bar_refresh_rate=1,
    lr=1e-2,
    n_epochs_kl_warmup=400,
    plan_kwargs=dict(
        loss_mask=test_mask,
        myparameters=myparameters,
    ),
)
rec_loss, rec_loss_all = spatial_model.get_metric()

# # spatial_model.save(mdl_path)
# # pass_results = pd.DataFrame(spatial_model.history).assign(
# #     heldout=heldout, lamb=lamb, train_phase=False
# # )
# # cv_results = cv_results.append(pass_results, ignore_index=True)
# # cv_results.to_pickle(os.path.join(WORKING_DIR, "spatial_cv.pickle"))

# # rec_loss, rec_loss_all = spatial_model.get_metric()
# # gene_infos = pd.DataFrame(
# #     {
# #         "gene": ["full"] + list(np.arange(len(rec_loss_all))),
# #         "reconstruction": [rec_loss] + list(rec_loss_all),
# #     }
# # ).assign(heldout=heldout, lamb=lamb, train_phase=False)
# # cv_results_metrics = cv_results_metrics.append(gene_infos, ignore_index=True)
