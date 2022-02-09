#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# path_sc = "/home/ubuntu/simulation_LN/sc_simu.h5ad"
# path_st = "/home/ubuntu/simulation_LN/st_simu.h5ad"
# params are 
# ID clustering
# path in
# path out
path_in <- args[1] 
dir_out <- args[2]
index_key = args[3]

path_sc <- paste(path_in, "/sc_simu.h5ad", sep="")
path_st = paste(path_in, "/st_simu.h5ad", sep="")
path_out = paste(path_in, dir_out, sep="")

library(RCTD)   
library(Matrix)
library(data.table)
library(Seurat)
# library(tidyverse)
library("reticulate")
sc <- import("scanpy")
pd <- import("pandas")

#' Transpose a dgRMatrix and simultaneously convert it to dgCMatrix
#' @param inmat input matrix in dgRMatrix format
#' @return A dgCMatrix that is the transposed dgRMatrix
#' @export transpose_dgRMatrix
transpose_dgRMatrix <- function(inmat) {
  if(class(inmat) != 'dgRMatrix')
    stop('inmat is not of class dgRMatrix')
  out <- new('dgCMatrix',
             i=inmat@j,
             p=inmat@p,
             x=inmat@x,
             Dim=rev(inmat@Dim),
             Dimnames=rev(inmat@Dimnames)
  )
  out
}      

# load single-cell data
adata <- sc$read(path_sc)
data = transpose_dgRMatrix(adata$X)
colnames(data) <- rownames(adata$obs)
rownames(data) <- rownames(adata$var)
single_cell <- CreateSeuratObject(counts = data)
key = adata$uns["key_clustering"][as.integer(index_key) + 1]
print(paste("Running RCTD on key:", key))

# add cell types
meta = adata$obs[[key]]
single_cell <- AddMetaData(
  object = single_cell,
  metadata = meta,
  col.name = 'cell_type'
)

# load spatial data
adata_st <- sc$read(path_st)
data_st = transpose_dgRMatrix(adata_st$X)
colnames(data_st) <- rownames(adata_st$obs)
rownames(data_st) <- rownames(adata_st$var)
# get vignette data format
datadir <- system.file("extdata",'SpatialRNA/Vignette',package = 'RCTD') # directory for sample Slide-seq dataset
spatial <- read.SpatialRNA(datadir) # read in the SpatialRNA object
# stuff our info inside of that
spatial@counts = data_st
coords = as.data.frame(adata_st$obsm["locations"])
colnames(coords) = c('xcoord', 'ycoord')
rownames(coords) = rownames(adata_st$obs)
spatial@coords = coords
spatial@nUMI <- adata_st$obs$n_counts
names(spatial@nUMI) <- rownames(adata_st$obs)
spatial@cell_type_names = as.character(unique(meta))
spatial@n_cell_type=length(unique(meta))

# the column for annotations is hard-coded
single_cell$liger_ident_coarse = factor(as.character(meta))
names(single_cell$liger_ident_coarse) <- rownames(adata$obs)
single_cell@meta.data$nUMI = adata$obs$n_counts
names(single_cell@meta.data$nUMI) <- rownames(adata$obs)
single_cell@meta.data$orig.ident = meta
names(single_cell@meta.data$orig.ident) <- rownames(adata$obs)
single_cell@meta.data$cluster = meta
names(single_cell@meta.data$orig.ident) <- rownames(adata$obs)


myRCTD <- create.RCTD(spatial, single_cell, max_cores = 1, CELL_MIN_INSTANCE = 1)
myRCTD <- run.RCTD(myRCTD, doublet_mode = FALSE)

results <- myRCTD@results
# normalize the cell type proportions to sum to 1.
norm_weights = sweep(results$weights, 1, rowSums(results$weights), '/') 
norm_weights = as.data.frame(norm_weights)
rownames(norm_weights) <- rownames(results$weights)
colnames(norm_weights) <- colnames(results$weights)

dir.create(path_out)
write.csv(norm_weights, paste(path_out, "output_weights.csv", sep=""))
write.csv(myRCTD@cell_type_info$renorm[[1]], paste(path_out, "output_dict.csv", sep=""))
