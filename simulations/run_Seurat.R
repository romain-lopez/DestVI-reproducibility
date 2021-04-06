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
print(paste("Running Seurat on key:", key))

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

anchors <- FindTransferAnchors(reference = sc_seu, query = sp_seu, normalization.method = "SCT", reduction='cca')
predictions.assay <- TransferData(anchorset = anchors, refdata = factor(sc_seu$cell_type),
                                  weight.reduction = 'cca')

decon_mtrx <- as.data.frame(predictions.assay[ , order(names(predictions.assay))][2:(length(predictions.assay)-1)])
rownames(decon_mtrx) = colnames(sp_seu)

dir.create(path_out)
write.csv(decon_mtrx, paste(path_out, "output_weights.csv", sep=""))