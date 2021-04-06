#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that creates the groundtruth for simulating data from the lymph node pairs.

Created on 2020/02/11
@author romain_lopez
"""

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numba import jit
import anndata
from sklearn.decomposition import SparsePCA, NMF, PCA
import scvi
import torch
scvi.settings.reset_logging_handler()
import logging
logger = logging.getLogger("scvi")

G = 2000
D = 4
param_path = "input_data/"


def cleanup_anndata(nova_data, G=G):
    # STEP1: remove some cell types
    mapping = {"Mature B cells": "B cells", 
               "Ifit3-high B cells": "B cells",
               "Cycling B/T cells": "B cells",
                "Plasma B cells": "NA",
            "Neutrophils": "NA",
               "Ly6-high monocytes": "NA",
               "Cxcl9-high monocytes": "NA",
              "Macrophages": "NA",
              "cDC1s":"NA",
               "NK cells":"NA",
               "GD T cells":"NA",
               "cDC2s":"NA",
               "Monocytes":"NA",
               "pDCs":"NA"
              }

    res = []
    for x in nova_data.obs["cell_types"]:
        local = x
        if x in mapping:
            local = mapping[x]
        res.append(local)

    nova_data.obs["broad_cell_types"] = res
    nova_data = nova_data[nova_data.obs["broad_cell_types"] != "NA"].copy()
    # STEP2: filter genes
    sc.pp.filter_genes(nova_data, min_counts=10)

    nova_data.layers["counts"] = nova_data.X.copy()

    sc.pp.highly_variable_genes(
        nova_data,
        n_top_genes=G,
        subset=True,
        layer="counts",
        flavor="seurat_v3"
    )
    return nova_data
    

def build_parameters_from_scrna_seq(nova_data):
   
    # learn dictionary of intra cell type variation
    sc.pp.normalize_total(nova_data, target_sum=10e4)
    sc.pp.log1p(nova_data)
    nova_data.raw = nova_data
    transformer = {}
    for t, ct in enumerate(np.unique(nova_data.obs["broad_cell_types"])):
        print(t, ct)
        transformer[t] = SparsePCA(n_components=4, random_state=0, alpha=5)
        transformer[t].fit(nova_data.X[nova_data.obs["broad_cell_types"] == ct].A[:, :])

    n_celltypes = np.unique(nova_data.obs["broad_cell_types"]).shape[0]
    n_genes = transformer[0].components_.shape[1]
    n_latent = transformer[0].components_.shape[0]
    mean_ = np.zeros(shape=(n_celltypes, n_genes))
    components_ = np.zeros(shape=(n_celltypes, n_latent, n_genes))
    for t in transformer.keys():
        mean_[t] = transformer[t].mean_
        components_[t] = transformer[t].components_
        
    return mean_, components_



if __name__ == '__main__':
    nova_data = sc.read_h5ad(path + "nova_final_data.h5ad")
    nova_data = cleanup_anndata(nova_data, G=G)
    # STEP 1: learn scVI to get dispersions
    scvi.data.setup_anndata(nova_data, layer="counts")
    model = scvi.model.SCVI(nova_data, n_latent=10)
    model.train(max_epochs=15)
    dispersion = torch.exp(model.module.px_r.detach()).cpu().numpy()
    np.save(path+ "inv-dispersion.npy", dispersion)
    # STEP 2: load Stereoscope parameters to get multiplicative factors
    # np.save(path + "beta_stereoscope.npy", spatial_model.module.beta.detach().cpu().numpy())
    # STEP 3: learn sparsePCA for intra-cell type variations
    mean_, components_ = build_parameters_from_scrna_seq(nova_data, G=G)
    np.savez(path + 'grtruth_PCA.npz', mean_=mean_, components_=components_)

