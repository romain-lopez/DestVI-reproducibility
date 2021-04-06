#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that runs DestVI on simulated data.

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

import scvi
from scvi.model import CondSCVI, DestVI

scvi.settings.reset_logging_handler()
import logging
logger = logging.getLogger("scvi")

@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input gene expression directory')
@click.option("--output-suffix", type=click.STRING, default="destvi", help="subdirectory for output saved model")
@click.option('--sc-epochs', type=click.INT, default=15, help='Max epochs for sc-rna')
@click.option('--st-epochs', type=click.INT, default=2000, help='Max epochs for st-rna')
@click.option('--amortization', type=click.STRING, default="latent", help="amortization mode for DestVI")
def main(input_dir, output_suffix, sc_epochs, st_epochs, amortization):
    # directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    #load data
    sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    logger.info("Running DestVI")

    # path management
    output_dir = input_dir + output_suffix + "_" + amortization + '/'

    if not os.path.isdir(output_dir):
        logger.info("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    else:
        logger.info(F"Found directory at:{output_dir}")

    # setup ann data
    scvi.data.setup_anndata(sc_adata, labels_key="cell_type")
    mapping = sc_adata.uns["_scvi"]["categorical_mappings"]["_scvi_labels"]["mapping"]

    # train sc-model
    sc_model = CondSCVI(sc_adata, n_latent=4, n_layers=2, n_hidden=128)
    sc_model.train(max_epochs=sc_epochs, plan_kwargs={"n_epochs_kl_warmup":2}, progress_bar_refresh_rate=0)
    plt.plot(sc_model.history["elbo_train"], label="train")
    plt.title("ELBO on train set over training epochs")
    plt.legend()
    plt.savefig(output_dir + "sc_model_training.png")
    plt.clf()
    logger.info(F"Last computed scELBO TRAIN: {sc_model.history['elbo_train'].iloc[-1]}")

    scvi.data.setup_anndata(st_adata)

    spatial_model = DestVI.from_rna_model(st_adata, sc_model, amortization=amortization, vamp_prior_p=100)
    spatial_model.train(max_epochs=st_epochs, train_size=1., plan_kwargs={"lr":0.001}, progress_bar_refresh_rate=0)

    plt.plot(spatial_model.history["elbo_train"][150:], label="train")
    plt.title("ELBO train over training epochs")
    plt.legend()
    plt.savefig(output_dir + "st_model_training.png")
    plt.clf()
    logger.info(F"Last computed stELBO TRAIN: {spatial_model.history['elbo_train'].iloc[-1]}")

    spatial_model.save(output_dir, overwrite=True)

if __name__ == '__main__':
    main()