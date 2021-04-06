#!/usr/bin/env python3
# -*- coding: utf-8

"""
Script that runs cell2location on simulated data, sent from Vitalii

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
data_type = 'float32'
import cell2location

@click.command()
@click.option('--input-dir', type=click.STRING, default="out/", help='input gene expression directory')
@click.option("--output-suffix", type=click.STRING, default="cell2location", help="subdirectory for output saved model")
@click.option('--index-key', type=click.INT, default=0, help='Which cell clustering to work with')
def main(input_dir, output_suffix, index_key):
    # directory management
    if input_dir[-1] != "/":
        input_dir += "/"
    #load data
    sc_adata = sc.read_h5ad(input_dir + "sc_simu.h5ad")
    st_adata = sc.read_h5ad(input_dir + "st_simu.h5ad")

    key = sc_adata.uns["key_clustering"][index_key]
    print(F"Running cell2location on key: {key}")

    # path management
    output_dir = input_dir + output_suffix + str(index_key) + '/'

    if not os.path.isdir(output_dir):
        print("Directory doesn't exist, creating it")
        os.mkdir(output_dir)
    else:
        print(F"Found directory at:{output_dir}")

    sc_adata.X = sc_adata.X.A
    st_adata.X = st_adata.X.A
    sc_adata.raw = sc_adata
    st_adata.raw = st_adata   


    r = cell2location.run_cell2location(

        # Single cell reference signatures as pd.DataFrame
        # (could also be data as anndata object for estimating signatures
        #  as cluster average expression - `sc_data=adata_snrna_raw`)
        sc_data=sc_adata,
        # Spatial data as anndata object
        sp_data=st_adata,

        # the column in sc_data.obs that gives cluster idenitity of each cell
        summ_sc_data_args={'cluster_col': key,
                            # select marker genes of cell types by specificity of their expression signatures
                            'selection': "cluster_specificity",
                            # specificity cutoff (1 = max, 0 = min)
                            'selection_specificity': 0.07
                            },

        train_args={'use_raw': True, # By default uses raw slots in both of the input datasets.
                    'n_iter': 40000, # Increase the number of iterations if needed (see QC below)

                    # Whe analysing the data that contains multiple experiments,
                    # cell2location automatically enters the mode which pools information across experiments
                    # 'sample_name_col': 'sample'
                    }, # Column in sp_data.obs with experiment ID (see above)


        export_args={'path': output_dir + "log", # path where to save results
                    'run_name_suffix': '' # optinal suffix to modify the name the run
                    },

        model_kwargs={ # Prior on the number of cells, cell types and co-located groups

                        'cell_number_prior': {
                            # - N - the expected number of cells per location:
                            'cells_per_spot': 8,
                            # - A - the expected number of cell types per location:
                            'factors_per_spot': 9,
                            # - Y - the expected number of co-located cell type groups per location
                            'combs_per_spot': 5
                        },

                        # Prior beliefs on the sensitivity of spatial technology:
                        'gene_level_prior':{
                            # Prior on the mean
                            'mean': 1/2,
                            # Prior on standard deviation,
                            # a good choice of this value should be at least 2 times lower that the mean
                            'sd': 1/4
                        }
        }
    )

if __name__ == '__main__':
    main()