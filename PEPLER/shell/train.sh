#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./shell/train.sh [GPU_ID] [modlora|moddora|modadalora|lora|dora|adalora]"
    echo "Example: ./shell/train.sh 0 modadalora"
    exit 0
fi

GPU_ID="${1:-0}"
MODEL_TYPE="${2:-modlora}"

case "$MODEL_TYPE" in
    modlora|moddora|modadalora|lora|dora|adalora) ;;
    *)
        echo "Error: unsupported model type '$MODEL_TYPE'. Choose modlora, moddora, modadalora, lora, dora, or adalora." >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SEED=111
MOD_RANK=8
BASE_RANK=$((3 * MOD_RANK))
ADALORA_INIT_FACTOR=3
case "$MODEL_TYPE" in
    lora|dora|adalora) RANK="$BASE_RANK" ;;
    modlora|moddora|modadalora) RANK="$MOD_RANK" ;;
esac
if [[ "$MODEL_TYPE" == "adalora" ]]; then
    ADALORA_TARGET_R="$BASE_RANK"
else
    ADALORA_TARGET_R="$MOD_RANK"
fi
ADALORA_INIT_R=$((ADALORA_INIT_FACTOR * ADALORA_TARGET_R))
ADALORA_TINIT=0
ADALORA_TFINAL=0
ADALORA_DELTA_T=1
ADALORA_ORTH_REG_WEIGHT=0.5
if [[ "$MODEL_TYPE" == "adalora" || "$MODEL_TYPE" == "modadalora" ]]; then
    model_name="${MODEL_TYPE}_llama_init-${ADALORA_INIT_R}_target-${ADALORA_TARGET_R}"
else
    model_name="${MODEL_TYPE}_llama_rank-${RANK}"
fi
data_root=../data
LLM_MODEL=/root/autodl-fs/Qwen2.5-7B/
CLIP_MODEL=../llms/clip-vit-base-patch32/
checkpoint_root=./checkpoints
out_file_root=./outputs
out_log_root=./logs
out_file_name=generated.txt
out_log_name=train.log
case "$MODEL_TYPE" in
    lora|modlora) LR=2e-7 ;;
    dora|moddora|adalora|modadalora) LR=2e-5 ;;
esac
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

for d_type in ClothingShoesAndJewelry MoviesAndTV; do
    for d_index in 1 2 3 4 5; do
        run_name="${d_index}/${model_name}"
        log_dir="${out_log_root}/${d_type}/${run_name}"
        output_dir="${out_file_root}/${d_type}/${run_name}"
        checkpoint_dir="${checkpoint_root}/${d_type}/${model_name}"
        mkdir -p "$log_dir" "$output_dir" "$checkpoint_dir"

        echo "GPU: ${GPU_ID}, model: ${MODEL_TYPE}, dataset: ${d_type}/${d_index}, seed: ${SEED}, lr: ${LR}, lora_modules: ${LoRA_MODULES}, run: ${model_name}"
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
            -adalora_init_r "$ADALORA_INIT_R" \
            -adalora_target_r "$ADALORA_TARGET_R" \
            -adalora_tinit "$ADALORA_TINIT" \
            -adalora_tfinal "$ADALORA_TFINAL" \
            -adalora_delta_t "$ADALORA_DELTA_T" \
            -adalora_orth_reg_weight "$ADALORA_ORTH_REG_WEIGHT" \
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
