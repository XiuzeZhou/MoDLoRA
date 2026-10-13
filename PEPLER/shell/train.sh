#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./shell/train.sh [GPU_ID] [modlora|lora|dora]"
    echo "Example: ./shell/train.sh 0 modlora"
    exit 0
fi

GPU_ID="${1:-0}"
MODEL_TYPE="${2:-dora}"

case "$MODEL_TYPE" in
    modlora|lora|dora) ;;
    *)
        echo "Error: unsupported model type '$MODEL_TYPE'. Choose modlora, lora, or dora." >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SEED=111
RANK=8
model_name="${MODEL_TYPE}_qwen_rank-${RANK}"
data_root=../data
LLM_MODEL=/root/autodl-fs/Qwen2.5-7B/
CLIP_MODEL=../llms/clip-vit-base-patch32/
checkpoint_root=./checkpoints
out_file_root=./outputs
out_log_root=./logs
out_file_name=generated.txt
out_log_name=train.log
LR=2e-7
K=768
EPOCHS=1
BATCH_SIZE=20
RATING_REG=0.01
LoRA_MODULES=7
MLP_SIZE=400
UI_SCALING=2.0
IMAGE_SCALING=2.0
ACC_STEPS=1
PYTHON_EXE="/root/miniconda3/bin/python"

for d_type in ClothingShoesAndJewelry MoviesAndTV
do
    for d_index in 1 2 3 4 5
    do
        run_name="${d_index}/${model_name}"
        log_dir="${out_log_root}/${d_type}/${run_name}"
        output_dir="${out_file_root}/${d_type}/${run_name}"
        checkpoint_dir="${checkpoint_root}/${d_type}/${model_name}"
        mkdir -p "$log_dir" "$output_dir" "$checkpoint_dir"

        echo "GPU: ${GPU_ID}, model: ${MODEL_TYPE}, dataset: ${d_type}/${d_index}, seed: ${SEED}, lora_modules: ${LoRA_MODULES}, rank: ${RANK}"
        TRANSFORMERS_CACHE=../llms/ \
        HF_DATASETS_CACHE=../llms/ \
        CUDA_VISIBLE_DEVICES="$GPU_ID" "${PYTHON_EXE}" -u ./main.py \
            -data_path "${data_root}/${d_type}/reviews.pickle" \
            -index_dir "${data_root}/${d_type}/${d_index}/" \
            -llm_model "$LLM_MODEL" \
            -clip_model "$CLIP_MODEL" \
            -lr "$LR" \
            -epochs "$EPOCHS" \
            -batch_size "$BATCH_SIZE" \
            -rating_reg "$RATING_REG" \
            -mlp_size "$MLP_SIZE" \
            -k "$K" \
            -r "$RANK" \
            -lora_modules "$LoRA_MODULES" \
            -ui_multimodal_scale "$UI_SCALING" \
            -image_multimodal_scale "$IMAGE_SCALING" \
            -acc_steps "$ACC_STEPS" \
            -seed "$SEED" \
            -cuda \
            -log_interval 200 \
            -checkpoint "$checkpoint_dir" \
            -outf "${output_dir}/${out_file_name}" \
            -words 20 \
            -model_type "$MODEL_TYPE" \
            > "${log_dir}/${out_log_name}" 2>&1
    done
done
