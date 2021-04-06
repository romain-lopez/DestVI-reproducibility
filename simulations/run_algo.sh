#!/bin/bash
export PYTHONUNBUFFERED=1

# Step 1: make dataset
# python make_dataset.py --output-dir /home/ubuntu/simu_runs/run_B


# Step 2: run algorithms
# python -u run_destVI.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --amortization latent
# python -u run_destVI.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --amortization none
# python -u run_destVI.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --amortization proportion
# python -u run_destVI.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --amortization both

# python -u run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --index-key 0
# python -u run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --index-key 1
# python -u run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --index-key 2
# python -u run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_B --sc-epochs 15 --st-epochs 2500 --index-key 3

# python -u run_embedding.py --input-dir /home/ubuntu/simu_runs/run_B --output-suffix harmony --algorithm Harmony
# python -u run_embedding.py --input-dir /home/ubuntu/simu_runs/run_B --output-suffix scanorama --algorithm Scanorama
# python -u run_embedding.py --input-dir /home/ubuntu/simu_runs/run_B --output-suffix scvi --algorithm scVI

# Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_B /RCTD0/ 0
# Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_B /RCTD1/ 1
# Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_B /RCTD2/ 2
# Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_B /RCTD3/ 3

# Rscript --vanilla run_spotlight.R  /home/ubuntu/simu_runs/run_B /spotlight0/ 0
# Rscript --vanilla run_spotlight.R  /home/ubuntu/simu_runs/run_B /spotlight1/ 1
# Rscript --vanilla run_spotlight.R  /home/ubuntu/simu_runs/run_B /spotlight2/ 2
# Rscript --vanilla run_spotlight.R  /home/ubuntu/simu_runs/run_B /spotlight3/ 3

# Rscript --vanilla run_Seurat.R  /home/ubuntu/simu_runs/run_B /seurat0/ 0
# Rscript --vanilla run_Seurat.R  /home/ubuntu/simu_runs/run_B /seurat1/ 1
# Rscript --vanilla run_Seurat.R  /home/ubuntu/simu_runs/run_B /seurat2/ 2
# Rscript --vanilla run_Seurat.R  /home/ubuntu/simu_runs/run_B /seurat3/ 3

# python -u run_cell2location.py --input-dir /home/ubuntu/simu_runs/run_B --index-key 0


# Step 3: eval methods
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir destvi_latent --model-string DestVI_latent
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir destvi_none --model-string DestVI_none
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir destvi_both --model-string DestVI_both
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir destvi_proportion --model-string DestVI_proportion


python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir stereo0 --model-string Stereoscope0
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir stereo1 --model-string Stereoscope1
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir stereo2 --model-string Stereoscope2
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir stereo3 --model-string Stereoscope3

python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir harmony --model-string Harmony
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir scanorama --model-string Scanorama
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir scvi --model-string scVI

python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir RCTD0 --model-string RCTD0
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir RCTD1 --model-string RCTD1
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir RCTD2 --model-string RCTD2
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir RCTD3 --model-string RCTD3

python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir seurat0 --model-string Seurat0
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir seurat1 --model-string Seurat1
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir seurat2 --model-string Seurat2
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir seurat3 --model-string Seurat3

python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir spotlight0 --model-string Spotlight0
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir spotlight1 --model-string Spotlight1
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir spotlight2 --model-string Spotlight2
# python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_B --model-subdir spotlight3 --model-string Spotlight3
