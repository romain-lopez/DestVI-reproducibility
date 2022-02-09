# load python and packages
library(Seurat)
library(reticulate)
ad <- import("anndata")
sc <- import("scanpy")

# import data
seurat_object = readRDS("destVI-paper-code/scope-seq-liver/Liver_normal_10um_annotated.rds")
# filter genes
variable_genes = VariableFeatures(seurat_object)
seurat_object <- seurat_object[variable_genes]

# try and reproduce seurat
# seurat_object <- NormalizeData(seurat_object, verbose = FALSE, 
#                                assay = "Spatial", normalization.method = "LogNormalize", scale.factor = 10000)
# seurat_object <- ScaleData(seurat_object, assay="Spatial")
# 
# seurat_object <- RunPCA(seurat_object, features=variable_genes, assay = "Spatial", verbose = TRUE)
# seurat_object <- RunUMAP(seurat_object, reduction = "pca", dims = 1:30, assay="Spatial")
# DimPlot(seurat_object, reduction = "umap", label = TRUE)

# try SCT
# seurat_object <- SCTransform(seurat_object, assay = "Spatial", verbose = TRUE)
# seurat_object <- RunPCA(seurat_object, features=variable_genes, assay = "SCT", verbose = TRUE)
# seurat_object <- RunUMAP(seurat_object, reduction = "pca", dims = 1:30, assay="SCT")
# DimPlot(seurat_object, reduction = "umap", label = TRUE)

# extract information
count_matrix = seurat_object@assays$Spatial@counts
# count_matrix = seurat_object@assays$SCT@data
meta_data = seurat_object@meta.data
umap = seurat_object@reductions$umap@cell.embeddings

meta_data["cell_type"] = seurat_object@active.ident

adata_seurat <- sc$AnnData(
  X   = t(as.matrix(count_matrix)), 
  obs = meta_data,
  var = GetAssay(seurat_object)[[]]
)

adata_seurat$write("/home/ubuntu/destVI-paper-code/scope-seq-liver/Liver_normal_10um_annotated.h5ad")

# below is just crazy stuff


genes = as.data.frame(rep(0, length(rownames(count_matrix))), row.names=rownames(count_matrix), col.names=("name"))


#' Transpose a dgCMatrix and simultaneously convert it to dgRMatrix
#' @param inmat input matrix in dgCMatrix format
#' @return A dgRMatrix that is the transposed dgCMatrix
#' @export transpose_dgRMatrix
transpose_dgCMatrix <- function(inmat) {
  if(class(inmat) != 'dgCMatrix')
    stop('inmat is not of class dgCMatrix')
  out <- new('dgRMatrix',
             j=inmat@i,
             p=inmat@p,
             x=inmat@x,
             Dim=rev(inmat@Dim),
             Dimnames=rev(inmat@Dimnames)
  )
  out
}      

adata_seurat <- sc$AnnData(
  X   = transpose_dgCMatrix(count_matrix),
  obs = meta_data,
  var = genes
)

