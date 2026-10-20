#!/bin/bash
set -e

# ==============================================================================
# 0. Run safety checks and GPU settings: ./shell/train.sh 0 dora
# ==============================================================================
if [ -z "$1" ]; then
  echo "Error: Please enter the number of the GPU!"
  echo "Usage: $0 <GPU_ID> [lora|modlora|dora|moddora|adalora|modadalora]"
  echo "Example: $0 0 dora"
  exit 1
fi

GPU_ID=$1
LORA_TYPE="${2:-modlora}"
echo "Using GPU: ${GPU_ID}"


# ==============================================================================
# 1. Environment variables and path configuration
# ==============================================================================
# Dataset: ClothingShoesAndJewelry, MoviesAndTV
DATASET_NAMES=("ClothingShoesAndJewelry" "MoviesAndTV")
SPLIT_INDICES="1,2,3,4,5"
DATA_DIR="../data/"
LOG_DIR="./logs/"
OUTPUT_DIR="./output/"


# ==============================================================================
# 2. Hyperparameter settings
# ==============================================================================
EPOCHS=3
LR=1e-3
ACC_STEPS=1
DELTA=0.2
WORD_LEN=20
ID_HIDDEN=1024
BATCH_SIZE=40
MOD_RANK=4
BASE_RANK=$((3 * MOD_RANK))
ADALORA_INIT_FACTOR=3
case "${LORA_TYPE}" in
    lora|dora|adalora) RANK="$BASE_RANK" ;;
    modlora|moddora|modadalora) RANK="$MOD_RANK" ;;
    *) echo "Invalid LORA_TYPE: ${LORA_TYPE}" >&2; exit 1 ;;
esac
ADALORA_ARGS=()
if [[ "${LORA_TYPE}" == "adalora" || "${LORA_TYPE}" == "modadalora" ]]; then
    ADALORA_TARGET_R="${RANK}"
    ADALORA_INIT_R=$((ADALORA_INIT_FACTOR * ADALORA_TARGET_R))
    ADALORA_ARGS=(--adalora_target_r "${ADALORA_TARGET_R}" --adalora_init_r "${ADALORA_INIT_R}")
fi
LORA_MODULES=2
CKPT_DIR="./checkpoints/"
MODEL_NAME="/root/autodl-fs/Qwen2.5-7B/"  # Qwen2.5-7B, Mistral-7B-Instruct-v0.3, gemma-7b, llama2-7B
CLIP_MODEL="../llms/clip-vit-base-patch32/"
PYTHON_EXE="/root/miniconda3/bin/python"
ADAPTER_ARGS=(--lora_type "${LORA_TYPE}")
case "${LORA_TYPE}" in
    lora) METHOD_NAME="CIER + LoRA" ;;
    modlora) METHOD_NAME="CIER + MoDLoRA" ;;
    dora) METHOD_NAME="CIER + DoRA" ;;
    moddora) METHOD_NAME="CIER + MoDDoRA" ;;
    adalora) METHOD_NAME="CIER + AdaLoRA" ;;
    modadalora) METHOD_NAME="CIER + MoDAdaLoRA" ;;
    *) echo "Invalid LORA_TYPE: ${LORA_TYPE}" >&2; exit 1 ;;
esac
case "${MODEL_NAME,,}" in
    *qwen*) LOG_NAME="train_qwen_${RANK}.log" ;;
    *mistral*) LOG_NAME="train_mistral_${RANK}.log" ;;
    *gemma*) LOG_NAME="train_gemma_${RANK}.log" ;;
    *llama*) LOG_NAME="train_llama_${RANK}.log" ;;
    *) echo "Cannot determine log name from MODEL_NAME: ${MODEL_NAME}" >&2; exit 1 ;;
esac
RUN_LOG_NAME="${LORA_TYPE}_${LOG_NAME}"

# ==============================================================================
# 3. run main.py
# ==============================================================================
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    IMAGE_EMBEDDING_PATH="${DATA_DIR}${DATASET_NAME}/embeddings_cache/item_embeddings.pt"

    if [ ! -f "${IMAGE_EMBEDDING_PATH}" ]; then
        echo "Image embedding cache not found for ${DATASET_NAME}. Preparing it first..."
        PREPARE_CMD=(
            "${PYTHON_EXE}" -u main.py
            --model_name "${MODEL_NAME}"
            --clip_model "${CLIP_MODEL}"
            --image_embedding_path "${IMAGE_EMBEDDING_PATH}"
            --dataset_name "${DATASET_NAME}"
            --data_dir "${DATA_DIR}"
            "${ADAPTER_ARGS[@]}"
            --use_multimodal
            --prepare_image_embeddings_only
        )
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PREPARE_CMD[@]}"
    fi

    echo "===================================================================="
    echo "Starting training process..."
    echo "Model:   ${MODEL_NAME}"
    echo "Method:  ${METHOD_NAME}"
    echo "Rank:    ${RANK}"
    echo "Dataset: ${DATASET_NAME}"
    echo "Splits:  ${SPLIT_INDICES}"
    echo "Log:     ${LOG_DIR}${DATASET_NAME}/${RUN_LOG_NAME}"
    echo "===================================================================="

    CMD=(
        "${PYTHON_EXE}" -u main.py
        --model_name "${MODEL_NAME}"
        --image_embedding_path "${IMAGE_EMBEDDING_PATH}"
        --dataset_name "${DATASET_NAME}"
        --split_indices "${SPLIT_INDICES}"
        --data_dir "${DATA_DIR}"
        --ckpt_dir "${CKPT_DIR}"
        --log_dir "${LOG_DIR}"
        --log_name "${LOG_NAME}"
        --output_dir "${OUTPUT_DIR}"
        --batch_size "${BATCH_SIZE}"
        --epochs "${EPOCHS}"
        --learning_rate "${LR}"
        --accumulation_steps "${ACC_STEPS}"
        --delta "${DELTA}"
        --word "${WORD_LEN}"
        --id_hidden "${ID_HIDDEN}"
        --lora_modules "${LORA_MODULES}"
        --r "${RANK}"
        "${ADAPTER_ARGS[@]}"
        "${ADALORA_ARGS[@]}"
        --use_multimodal
    )

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"

    echo "Process finished for ${DATASET_NAME}. Please check ${LOG_DIR}${DATASET_NAME}/${RUN_LOG_NAME}"
done
