#!/bin/bash
#SBATCH --job-name=cf_cbm
#SBATCH --output=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A.out
#SBATCH --error=/home/zhangsd/repos/CBM-MOF/slurm_logs/%x_%A.err
#SBATCH --partition=G4090          # c3 / RTX 4090 (default GPU partition)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --gres=gpu:1

# ── Environment ─────────────────────────────────────────────────────────────
export PATH=/opt/share/miniconda3/envs/crystalframer_env/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/crystalframer_env/lib/:$LD_LIBRARY_PATH

# nvidia-* pip packages install CUDA libs under site-packages/nvidia/*/lib/
_PYVER=3.11
_SP=/opt/share/miniconda3/envs/crystalframer_env/lib/python${_PYVER}/site-packages
for _d in "$_SP"/nvidia/*/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done

# ── Directories ────────────────────────────────────────────────────────────
CF_REPO=/home/zhangsd/repos/crystalframer
CBM_REPO=/home/zhangsd/repos/CBM-MOF
mkdir -p "$CBM_REPO/slurm_logs"

# ── Parse mode from first argument (default: dryrun) ─────────────────────────
MODE="${1:-dryrun}"

if [[ "$MODE" == "full" ]]; then
    EPOCHS=800
    BATCH_SIZE=64
    SWA=50
elif [[ "$MODE" == "short" ]]; then
    EPOCHS=100
    BATCH_SIZE=64
    SWA=10
else
    # dryrun: 10 epochs, small batch, single GPU
    EPOCHS=10
    BATCH_SIZE=32
    SWA=0
fi

echo "=== CrystalFramer CBM-MOF Training ==="
echo "  Mode       : $MODE"
echo "  Epochs     : $EPOCHS"
echo "  Batch size : $BATCH_SIZE"
echo "  Partition  : G4090 (c3 / RTX 4090)"
echo "  GPU(s)     : ${SLURM_JOB_GPUS:-auto-assigned by SLURM}"
echo "  Start time : $(date)"
echo "======================================="

# ── Step 1: Prepare data if not already done ─────────────────────────────────
PKL_TRAIN="$CF_REPO/data/cbm_mof/train/raw/raw_data.pkl"
if [[ ! -f "$PKL_TRAIN" ]]; then
    echo ">>> Preparing dataset (CIF → pkl)..."
    cd "$CBM_REPO"
    CUDA_VISIBLE_DEVICES="" python src/crystalframer/prepare_data_cf.py \
        --cif-dir data/processed/stratified_datasets/cifs \
        --output-dir "$CF_REPO/data/cbm_mof" \
        --splits-dir data/processed/stratified_datasets \
        --label-file src/ml/data/round2/RAC_and_zeo_features_with_id_prop.csv
    echo ">>> Data preparation done."
else
    echo ">>> Dataset already exists, skipping preparation."
fi

# ── Step 2: Train ─────────────────────────────────────────────────────────────
cd "$CF_REPO"
echo ">>> Starting training..."
# Use srun for proper SLURM GPU isolation and PL compatibility.
# SLURM sets CUDA_VISIBLE_DEVICES automatically via --gres=gpu:N.
srun python -u train.py \
    -p crystalframer/cbm_mof.json \
    --n_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --swa_epochs "$SWA" \
    --experiment_name "cbm_mof_${MODE}" \
    --save_path "result/cbm_mof_${MODE}"

echo "=== Finished at $(date) ==="
