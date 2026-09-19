echo $1
SEED=111
model_name=/modlora_qwen/
data_path=../data/
data_name=/reviews.pickle
LLM_MODEL=/root/autodl-fs/Qwen2.5-7B/
CLIP_MODEL=../llms/clip-vit-base-patch32/
checkpoint_dir=./checkpoints/
out_file_dir=./outputs/
out_file_name=generated.txt
out_log_dir=./logs/
out_log=train.log
LR=2e-7
K=768
RANK=8
EPOCHS=1
BATCH_SIZE=16
RATING_REG=0.01
LoRA_MODULES=7
MLP_SIZE=400
UI_SCALING=2.0
IMAGE_SCALING=2.0
ACC_STEPS=1

for d_type in ClothingShoesAndJewelry MoviesAndTV
do
    for d_index in 1 2 3 4 5
    do
        mkdir -p ${out_log_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${out_file_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${checkpoint_dir}${d_type}${model_name}

        echo "data_type: $d_type, data_index: $d_index , seed: $SEED, lora_modules: $LoRA_MODULES, r: $RANK"
        TRANSFORMERS_CACHE=../llms/ \
        HF_DATASETS_CACHE=../llms/ \
        CUDA_VISIBLE_DEVICES=$1 python -u ./main.py \
            -data_path ${data_path}${d_type}${data_name} \
            -index_dir ${data_path}${d_type}\/${d_index}\/ \
            -llm_model ${LLM_MODEL} \
            -clip_model ${CLIP_MODEL} \
            -lr $LR \
            -epochs $EPOCHS \
            -batch_size $BATCH_SIZE \
            -rating_reg $RATING_REG \
            -mlp_size $MLP_SIZE \
            -k $K \
            -r $RANK \
            -lora_modules $LoRA_MODULES \
            -ui_multimodal_scale $UI_SCALING \
            -image_multimodal_scale $IMAGE_SCALING \
            -acc_steps $ACC_STEPS \
            -seed $SEED \
            -cuda \
            -log_interval 200 \
            -checkpoint ${checkpoint_dir}${d_type} \
            -outf ${out_file_dir}${d_type}\/${d_index}${model_name}${out_file_name} \
            -words 20 \
            -model_type uiadapter \
            > ${out_log_dir}${d_type}\/${d_index}${model_name}${out_log}
    done
done
