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
path_sp = paste(path_in, "/st_simu.h5ad", sep="")
path_out = paste(path_in, dir_out, sep="")

library(SPOTlight)
library(Seurat)
library(dplyr)

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
sc_seu <- CreateSeuratObject(counts = data)
key = adata$uns["key_clustering"][as.integer(index_key) + 1]
print(paste("Running Spotlight on key:", key))

# add cell types
meta = adata$obs[[key]]
sc_seu <- AddMetaData(
  object = sc_seu,
  metadata = meta,
  col.name = 'cell_type'
)

# load spatial data
sp_adata <- sc$read(path_sp)
sp_data = transpose_dgRMatrix(sp_adata$X)
colnames(sp_data) <- rownames(sp_adata$obs)
rownames(sp_data) <- rownames(sp_adata$var)
sp_seu <- CreateSeuratObject(counts = sp_data)


sc_seu <- SCTransform(sc_seu, verbose = FALSE) %>% RunPCA(verbose = FALSE)
sp_seu <- SCTransform(sp_seu, verbose = FALSE) %>% RunPCA(verbose = FALSE)

Seurat::Idents(object = sc_seu) <- sc_seu@meta.data$cell_type

cluster_markers_all <- Seurat::FindAllMarkers(object = sc_seu, 
                                              assay = "RNA",
                                              slot = "data",
                                              verbose = TRUE, 
                                              only.pos = TRUE, 
                                              logfc.threshold = 1,
                                              min.pct = 0.9)
spotlight_ls <- spotlight_deconvolution(se_sc = sc_seu,
                                        counts_spatial = sp_seu@assays$RNA@counts,
                                        clust_vr = "cell_type",
                                        cluster_markers = cluster_markers_all,
                                        cl_n = 100, # 100 by default
                                        hvg = 5000,
                                        ntop = NULL,
                                        transf = "uv",
                                        method = "nsNMF",
                                        min_cont = 0.09)
decon_mtrx <- as.data.frame(spotlight_ls[[2]])
rownames(decon_mtrx) = colnames(sp_seu)
decon_mtrx <- decon_mtrx[1:(length(decon_mtrx)-1)]

dir.create(path_out)
write.csv(decon_mtrx, paste(path_out, "output_weights.csv", sep=""))