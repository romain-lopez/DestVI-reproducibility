#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that runs embedding algorithms on simulated data.

Created on 2020/02/03
@author romain_lopez
"""

import os
import click
import numpy as np

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata

import harmonypy as hm
from sklearn.decomposition import PCA
import scanorama
import scvi

scvi.settings.reset_logging_handler()
import logging
logger = logging.getLogger("scvi")


@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input gene expression directory')
@click.option('--algorithm', type=click.STRING, default="Harmony", help="which algorithm to run")
@click.option("--output-suffix", type=click.STRING, default="harmony", help="output saved model")
def main(input_dir, algorithm, output_suffix):
    # directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    #load data
    sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    logger.info(F"Running {algorithm}")

    # path management
    output_dir = input_dir + output_suffix + '/'

    if not os.path.isdir(output_dir):
        logger.info("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    else:
        logger.info(F"Found directory at:{output_dir}")

    if algorithm == "Harmony":
        dat1 = sc_adata.X.A
        dat2 = st_adata.X.A
        data_mat = PCA(n_components=25).fit_transform(np.log(1 + np.vstack([dat1, dat2])))
        meta_data = pd.DataFrame(data= dat1.shape[0] * ["b1"] + dat2.shape[0] * ["b2"], columns=["batch"])
        ho = hm.run_harmony(data_mat, meta_data, ["batch"])
        embedding1, embedding2 = ho.Z_corr.T[:dat1.shape[0]], ho.Z_corr.T[dat1.shape[0]:]
    elif algorithm == "Scanorama":
        integrated, genes = scanorama.integrate([sc_adata.X, st_adata.X], [sc_adata.var.index, st_adata.var.index], dimred=25)
        embedding1, embedding2 = integrated
    elif algorithm == "scVI":
        concat_adata = anndata.concat([sc_adata, st_adata], label="dataset")
        concat_adata.obs_names_make_unique()
        scvi.data.setup_anndata(concat_adata, batch_key="dataset")
        model = scvi.model.SCVI(concat_adata, n_latent=25)
        model.train(max_epochs=15)
        embedding1 = model.get_latent_representation(concat_adata[concat_adata.obs["dataset"] == "0"])
        embedding2 = model.get_latent_representation(concat_adata[concat_adata.obs["dataset"] == "1"])

    else:
        raise NotImplementedError("Only scVI, Harmony and Scanorama")
    
    np.savez_compressed(output_dir + 'embedding.npz', embedding_sc=embedding1, embedding_st=embedding2)

if __name__ == '__main__':
    main()