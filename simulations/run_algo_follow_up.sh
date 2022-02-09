#!/bin/bash
export PYTHONUNBUFFERED=1


# List of datasets:
# 1. More cell types
# 2. cell type removed in scRNA-seq
# 3. cell type removed in spatial
# 4. fewer reads in spatial
# 5. cell types per spot (maybe both can be combined?)


# Step 1: make dataset
export PATH_EXPERIMENT=/home/ubuntu/simu_runs/run_C
python make_dataset_extended.py --output-dir $PATH_EXPERIMENT --temp-ct=1 --bin-sampling=1

export PATH_EXPERIMENT=/home/ubuntu/simu_runs/run_D
python make_dataset_extended.py --output-dir $PATH_EXPERIMENT --temp-ct=0.5 --bin-sampling=0.5 --ct-study 0

export PATH_EXPERIMENT=/home/ubuntu/simu_runs/run_E
python make_dataset_extended.py --output-dir $PATH_EXPERIMENT --temp-ct=0.2 --bin-sampling=0.2 --ct-study 0

export PATH_EXPERIMENT=/home/ubuntu/simu_runs/run_F
python make_dataset_extended.py --output-dir $PATH_EXPERIMENT --temp-ct=0.3 --bin-sampling=0.5 --ct-study 0

export PATH_EXPERIMENT=/home/ubuntu/simu_runs/run_G
python make_dataset_extended.py --output-dir $PATH_EXPERIMENT --temp-ct=0.5 --bin-sampling=0.2 --ct-study 0

# Location of partial datasets after renaming
export PATH_EXPERIMENT=/home/ubuntu/simu_runs/run_missing_st
export PATH_EXPERIMENT=/home/ubuntu/simu_runs/run_missing_sc
# Step 2: run algorithms
python -u run_destVI.py --input-dir $PATH_EXPERIMENT --output-suffix destvi --sc-epochs 40 --st-epochs 2800 --amortization latent

python -u run_stereoscope.py --input-dir $PATH_EXPERIMENT --sc-epochs 15 --st-epochs 2500 --index-key 0
python -u run_stereoscope.py --input-dir $PATH_EXPERIMENT --sc-epochs 15 --st-epochs 2500 --index-key 1
python -u run_stereoscope.py --input-dir $PATH_EXPERIMENT --sc-epochs 15 --st-epochs 2500 --index-key 2
python -u run_stereoscope.py --input-dir $PATH_EXPERIMENT --sc-epochs 15 --st-epochs 2500 --index-key 3

Rscript --vanilla run_RCTD.R  $PATH_EXPERIMENT /RCTD0/ 0
Rscript --vanilla run_RCTD.R  $PATH_EXPERIMENT /RCTD1/ 1
Rscript --vanilla run_RCTD.R  $PATH_EXPERIMENT /RCTD2/ 2
Rscript --vanilla run_RCTD.R  $PATH_EXPERIMENT /RCTD3/ 3

# Step 3: eval methods
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir destvi_latent --model-string DestVI_latent

python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo0 --model-string Stereoscope0
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo1 --model-string Stereoscope1
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo2 --model-string Stereoscope2
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo3 --model-string Stereoscope3

python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD0 --model-string RCTD0
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD1 --model-string RCTD1
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD2 --model-string RCTD2
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD3 --model-string RCTD3


# Step 3: eval methods for missing cell types
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir destvi_latent --model-string DestVI_latent --missing-ct sc
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo0 --model-string Stereoscope0 --missing-ct sc
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo1 --model-string Stereoscope1 --missing-ct sc
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo2 --model-string Stereoscope2 --missing-ct sc
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir stereo3 --model-string Stereoscope3 --missing-ct sc

python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD0 --model-string RCTD0 --missing-ct sc
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD1 --model-string RCTD1 --missing-ct sc
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD2 --model-string RCTD2 --missing-ct sc
python -u benchmark_extended.py --input-dir $PATH_EXPERIMENT --model-subdir RCTD3 --model-string RCTD3 --missing-ct sc
