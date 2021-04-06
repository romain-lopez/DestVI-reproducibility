#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that runs Stereoscope on simulated data.

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
from scvi.external import RNAStereoscope, SpatialStereoscope

scvi.settings.reset_logging_handler()
import logging
logger = logging.getLogger("scvi")

@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input gene expression directory')
@click.option("--output-suffix", type=click.STRING, default="stereo", help="subdirectory for output saved model")
@click.option('--index-key', type=click.INT, default=0, help='Which cell clustering to work with')
@click.option('--sc-epochs', type=click.INT, default=15, help='Max epochs for sc-rna')
@click.option('--st-epochs', type=click.INT, default=2000, help='Max epochs for st-rna')
def main(input_dir, output_suffix, index_key, sc_epochs, st_epochs):
    # directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    #load data
    sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    key = sc_adata.uns["key_clustering"][index_key]
    logger.info(F"Running Stereoscope on key: {key}")

    # path management
    output_dir = input_dir + output_suffix + str(index_key) + '/'

    if not os.path.isdir(output_dir):
        logger.info("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    else:
        logger.info(F"Found directory at:{output_dir}")


    # setup ann data
    scvi.data.setup_anndata(sc_adata, labels_key=key)
    mapping = sc_adata.uns["_scvi"]["categorical_mappings"]["_scvi_labels"]["mapping"]

    # train sc-model
    sc_stereo = RNAStereoscope(sc_adata, )
    sc_stereo.train(lr=0.01, max_epochs=sc_epochs, progress_bar_refresh_rate=0)

    plt.plot(sc_stereo.history["elbo_train"][2:], label="train")
    plt.title("loss over training epochs")
    plt.legend()
    plt.savefig(output_dir + "sc_model_training.png")
    plt.clf()
    logger.info(F"Last computed scELBO TRAIN: {sc_stereo.history['elbo_train'].iloc[-1]}")

    scvi.data.setup_anndata(st_adata)

    st_stereo = SpatialStereoscope.from_rna_model(st_adata, sc_stereo)
    st_stereo.train(max_epochs=st_epochs, progress_bar_refresh_rate=0)
    plt.plot(st_stereo.history["elbo_train"][150:], label = "train")
    plt.title("loss over training epochs")
    plt.legend()
    plt.savefig(output_dir + "st_model_training.png")
    plt.clf()
    logger.info(F"Last computed stELBO TRAIN: {st_stereo.history['elbo_train'].iloc[-1]}")

    st_stereo.save(output_dir, overwrite=True)

if __name__ == '__main__':
    main()