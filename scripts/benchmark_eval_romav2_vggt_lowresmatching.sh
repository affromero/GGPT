#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1

# Force matcher to run on feedforward (resized) resolution, not full-res input.
export GGPT_MATCH_ON_FF_RES=1

MATCH_MODEL=romav2-base
FEEDFORWARD_MODEL=vggt-point
NUM_GPUS=4

SFM_DIR="outputs/benchmark_lowres/sfm_${FEEDFORWARD_MODEL}-${MATCH_MODEL}"
GGPT_DIR="outputs/benchmark_lowres/ggpt_${FEEDFORWARD_MODEL}-${MATCH_MODEL}_lrmatch"

# Optional on nodes where NCCL OFI backend fails to initialize:
# export NCCL_NET=Socket

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --rdzv-endpoint=localhost:2026 \
  sfm/run_benchmark_sfm.py \
  common_config.ddp=True \
  match_config.models="${MATCH_MODEL}" \
  feedforward_config.model="${FEEDFORWARD_MODEL}" \
  hydra.run.dir="${SFM_DIR}"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --rdzv-endpoint=localhost:2026 \
  ggpt/run_benchmark_ggpt.py \
  common_config.ddp=True \
  hydra.run.dir="${GGPT_DIR}"
