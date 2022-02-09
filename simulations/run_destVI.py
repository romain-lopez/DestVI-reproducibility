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
@click.option('--sc-epochs', type=click.INT, default=20, help='Max epochs for sc-rna')
@click.option('--st-epochs', type=click.INT, default=4000, help='Max epochs for st-rna')
@click.option('--amortization', type=click.STRING, default="latent", help="amortization mode for DestVI")
@click.option("--weight", type=click.INT, default=0, help="whether to reweight observation")
@click.option("--layer", type=click.INT, default=0, help="whether to use the custom layer")
@click.option("--n-latent", type=click.INT, default=4, help="n_latent")
@click.option("--n-layers", type=click.INT, default=2, help="n_layers")
@click.option("--n-hidden", type=click.INT, default=128, help="n_hidden")
@click.option("--n-epochs-kl-warmup", type=click.INT, default=2, help="n_epochs_kl_warmup")
@click.option("--vamp-prior-p", type=click.INT, default=100, help="vamp_prior_p")
@click.option("--lr", type=click.FLOAT, default=0.001, help="learning rate")
@click.option("--l1-scalar", type=click.FLOAT, default=0, help="l1 regularization on loadings")
def main(input_dir, output_suffix, sc_epochs, st_epochs, amortization, weight, layer, n_latent, n_layers, n_hidden, n_epochs_kl_warmup, vamp_prior_p, lr, l1_scalar):
    # directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    # load data
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
    layer_f = "counts" if layer > 0 else None
    scvi.data.setup_anndata(sc_adata, labels_key="cell_type", layer=layer_f)

    mapping = sc_adata.uns["_scvi"]["categorical_mappings"]["_scvi_labels"]["mapping"]

    flag_w = True if weight > 0 else False
    # train sc-model
    sc_model = CondSCVI(sc_adata, n_latent=n_latent, n_layers=n_layers, n_hidden=n_hidden, weight_obs=flag_w)
    sc_model.train(max_epochs=sc_epochs, plan_kwargs={"n_epochs_kl_warmup":n_epochs_kl_warmup}, progress_bar_refresh_rate=0)
    plt.plot(sc_model.history["elbo_train"], label="train")
    plt.title("ELBO on train set over training epochs")
    plt.legend()
    plt.savefig(output_dir + "sc_model_training.png")
    plt.clf()
    logger.info(F"Last computed scELBO TRAIN: {sc_model.history['elbo_train'].iloc[-1]}")

    scvi.data.setup_anndata(st_adata, layer=layer_f)

    spatial_model = DestVI.from_rna_model(st_adata, sc_model, amortization=amortization, vamp_prior_p=vamp_prior_p, l1_scalar=l1_scalar)
    spatial_model.train(max_epochs=st_epochs, train_size=1., plan_kwargs={"lr":lr}, progress_bar_refresh_rate=0)

    plt.plot(spatial_model.history["elbo_train"][150:], label="train")
    plt.title("ELBO train over training epochs")
    plt.legend()
    plt.savefig(output_dir + "st_model_training.png")
    plt.clf()
    logger.info(F"Last computed stELBO TRAIN: {spatial_model.history['elbo_train'].iloc[-1]}")

    spatial_model.save(output_dir, overwrite=True)

if __name__ == '__main__':
    main()