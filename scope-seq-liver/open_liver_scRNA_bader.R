library(Seurat)
library(reticulate)
ad <- import("anndata")
sc <- import("scanpy")

library(HumanLiver)
viewHumanLiver()

seurat_object = HumanLiverSeurat
variable_genes = VariableFeatures(seurat_object)
seurat_object <- seurat_object[variable_genes]

count_matrix = seurat_object@assays$RNA@counts
meta_data = seurat_object@meta.data


count_matrix <- as.matrix(count_matrix)
genes = as.data.frame(rep(0, length(rownames(count_matrix))), row.names=rownames(count_matrix), col.names=("name"))


adata_seurat <- sc$AnnData(
  X   = t(count_matrix), 
  obs = meta_data,
  var = genes
)

adata_seurat$write("/home/ubuntu/destVI-paper-code/high-resolution/Liver_Bader_annotated.h5ad")


