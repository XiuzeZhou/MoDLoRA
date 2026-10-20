import torch
import random
import os
import gc
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from inspect import signature
from tqdm import tqdm
from torch.optim import *
# from typing import Optional, Callable, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

from dataloader import *
from utils import *
from model import *
from peft import AdaLoraConfig, LoraConfig, get_peft_model, TaskType
from argparse import ArgumentParser

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

def seed_everything(seed=5254):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def _load_embedding_object(path):
    if path.endswith(('.pickle', '.pkl')):
        return pd.read_pickle(path)
    return load_torch_object(path, map_location='cpu')

def load_torch_object(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def parse_split_indices(split_indices):
    if split_indices is None or str(split_indices).strip() == "":
        return ['1', '2', '3', '4', '5']
    return [idx.strip() for idx in str(split_indices).split(',') if idx.strip()]

def _build_item_index_mapping(args):
    raw_path = os.path.join(args.data_dir, args.dataset_name, 'reviews.pickle')
    raw_dataset = pd.DataFrame(pd.read_pickle(raw_path))
    encoder = LabelEncoder()
    item_ids = encoder.fit_transform(raw_dataset['item'].tolist())

    raw_to_idx = {}
    idx_to_raw = {}
    for raw_item, item_idx in zip(raw_dataset['item'].tolist(), item_ids):
        item_idx = int(item_idx)
        raw_to_idx[raw_item] = item_idx
        raw_to_idx[str(raw_item)] = item_idx
        idx_to_raw[item_idx] = raw_item
    return raw_to_idx, idx_to_raw

def _remap_image_embedding_object(image_embeddings, raw_to_idx):
    if not isinstance(image_embeddings, dict):
        return image_embeddings

    for table_key in ['image_embeddings', 'image_embedding', 'item_embeddings', 'embeddings', 'features']:
        if table_key in image_embeddings and not isinstance(image_embeddings[table_key], (int, float, str)):
            image_embeddings = image_embeddings[table_key]
            break

    if not isinstance(image_embeddings, dict):
        return image_embeddings

    remapped = {}
    for item_key, feature in image_embeddings.items():
        item_idx = None
        if isinstance(item_key, int):
            item_idx = item_key
        elif item_key in raw_to_idx:
            item_idx = raw_to_idx[item_key]
        elif str(item_key) in raw_to_idx:
            item_idx = raw_to_idx[str(item_key)]

        if item_idx is not None:
            remapped[int(item_idx)] = feature
    return remapped if remapped else image_embeddings

def _find_item_image(image_dir, raw_item):
    stem = str(raw_item)
    exact_path = os.path.join(image_dir, stem)
    if os.path.exists(exact_path):
        return exact_path
    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(image_dir, stem + ext)
        if os.path.exists(image_path):
            return image_path
    return None

def build_multimodal_image_embeddings(args, item_num, device):
    if not args.use_multimodal and not getattr(args, "prepare_image_embeddings_only", False):
        return None

    raw_to_idx, idx_to_raw = _build_item_index_mapping(args)
    dataset_dir = os.path.join(args.data_dir, args.dataset_name)

    if args.image_embedding_path:
        cache_path = args.image_embedding_path
        cache_dir = os.path.dirname(cache_path) or '.'
    else:
        cache_dir = os.path.join(dataset_dir, 'embeddings_cache')
        cache_path = os.path.join(cache_dir, 'item_embeddings.pt')
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading cached item image embeddings from {cache_path}")
        image_embeddings = load_torch_object(cache_path, map_location='cpu')
        image_embeddings = _remap_image_embedding_object(image_embeddings, raw_to_idx)
        return MyModel.build_image_embedding_table(image_embeddings, item_num)

    if args.clip_model is None:
        raise FileNotFoundError(
            f"Image embedding cache not found: {cache_path}. "
            "Run once with --clip_model and --prepare_image_embeddings_only to build it."
        )

    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("PIL and transformers CLIP classes are required for image embeddings.") from exc

    clip_device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    image_dir = os.path.join(dataset_dir, args.image_dir)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(clip_device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    for param in clip_model.parameters():
        param.requires_grad = False

    item_embeddings = {}
    print(f"Generating item image embeddings with CLIP for {len(idx_to_raw)} items")
    for item_idx, raw_item in tqdm(idx_to_raw.items()):
        image_path = _find_item_image(image_dir, raw_item)
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (300, 300), (255, 255, 255))

        image_inputs = clip_processor(images=image, return_tensors="pt").to(clip_device)
        with torch.no_grad():
            item_embeddings[item_idx] = clip_model.get_image_features(**image_inputs).cpu()

    torch.save(item_embeddings, cache_path)
    print(f"Item image embeddings saved to {cache_path}")
    del clip_model
    del clip_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return MyModel.build_image_embedding_table(item_embeddings, item_num)

def configure_gradient_checkpointing(model_llm, lora_type):
    model_llm.config.use_cache = False
    if lora_type in ('modlora', 'moddora', 'modadalora'):
        try:
            model_llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            print(f"Non-reentrant gradient checkpointing is not supported; disabling it for {lora_type}.")
            model_llm.gradient_checkpointing_disable()
    else:
        model_llm.gradient_checkpointing_enable()

def prepare_model_for_generation(model):
    if hasattr(model.model, "gradient_checkpointing_disable"):
        model.model.gradient_checkpointing_disable()
    if hasattr(model.model, "config"):
        model.model.config.use_cache = True

def load_adapter_state_dict(path, map_location="cpu"):
    return load_torch_object(path, map_location=map_location)

def cleanup_split():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def reset_peak_vram_stats(device):
    """Start a new peak-VRAM measurement window for the selected GPU."""
    if torch.cuda.is_available():
        # The model has already initialized this CUDA context before this helper
        # is called. Using the current device avoids allocator errors seen with
        # an explicit integer device on some PyTorch/CUDA combinations.
        with torch.cuda.device(device):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

def log_vram_stats(device, log_name, include_peak=False):
    """Print CUDA memory usage and append the same information to the run log."""
    if not torch.cuda.is_available():
        lines = ["[VRAM Profiling] CUDA is not available; memory profiling skipped."]
    else:
        with torch.cuda.device(device):
            torch.cuda.synchronize()
            lines = []
            if include_peak:
                peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                lines.extend([
                    "=" * 80,
                    f"[VRAM Profiling] Peak GPU Memory Allocated: {peak_memory_allocated:.2f} GB",
                    f"[VRAM Profiling] Peak GPU Memory Reserved (Cached): {peak_memory_reserved:.2f} GB",
                    "=" * 80,
                ])
            else:
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                lines.extend([
                    f"GPU memory allocated: {memory_allocated:.2f} GB",
                    f"GPU memory cached: {memory_reserved:.2f} GB",
                ])

    message = "\n".join(lines)
    print(message)
    with open(log_name, 'a+', encoding='utf-8') as log_file:
        log_file.write(message + "\n")

def log_trainable_parameters_summary(model, log_name):
    """Print and log the model's total and trainable parameter counts."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_ratio = (
        100.0 * trainable_parameters / total_parameters if total_parameters else 0.0
    )
    lines = [
        "--- Trainable Parameters Summary (LLM base) ---",
        f"Total parameters: {total_parameters:,}",
        f"Trainable parameters (LoRA + Embeddings): {trainable_parameters:,}",
        f"Trainable ratio: {trainable_ratio:.2f}%",
    ]
    message = "\n".join(lines)
    print(message)
    with open(log_name, 'a+', encoding='utf-8') as log_file:
        log_file.write(message + "\n")

def adalora_orthogonal_regularization(model):
    """Add the AdaLoRA factor regularizer to CIER's externally computed loss."""
    penalties = []
    for name, parameter in model.model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'lora_A' in name:
            factor = parameter.float()
            covariance = factor @ factor.T
        elif 'lora_B' in name:
            factor = parameter.float()
            covariance = factor.T @ factor
        else:
            continue
        identity = torch.eye(covariance.size(0), device=covariance.device)
        penalties.append(torch.linalg.matrix_norm(covariance - identity))
    return torch.stack(penalties).mean() if penalties else 0.0


def train_step(model,
               train_dataloader,
               optimizer,
               device,
               epoch, 
               show_train_loss_steps,
               accumulation_steps,
               log_name,
               global_step=0,
               rank_update=None,
               orth_reg_weight=0.0,
               ):
    model.train()
    optimizer.zero_grad()
    loss_log = []
    for batch_idx, [input_ids, userid, itemid, rating, curr_flag,
                    rating_inputs
                   ] in enumerate(tqdm(train_dataloader)):
        input_ids = input_ids.to(device)
        itemid = itemid.to(device)
        userid = userid.to(device)
        rating = rating.to(device)
        curr_flag = curr_flag.to(device)
        rating_inputs = rating_inputs.to(device)
        with autocast(dtype=torch.bfloat16):
            loss = model.train_step(input_ids, userid, itemid, rating, curr_flag,
                                    rating_inputs
                                   )
        if orth_reg_weight:
            loss = loss + orth_reg_weight * adalora_orthogonal_regularization(model)
        loss_log.append(loss.item())
        group_start = (batch_idx // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(train_dataloader) - group_start)
        loss = loss / group_size
        
        loss.backward() 
        
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(train_dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step() 
            global_step += 1
            if rank_update is not None:
                rank_update(global_step)
            optimizer.zero_grad()
        
        if (batch_idx + 1) % show_train_loss_steps == 0:
            f = open(log_name,'a+')
            f.write("Train Epoch: {} [{}/{} ({}%)]\t Loss: {}\n".format(epoch,
                                            (batch_idx + 1) * input_ids.shape[0],
                                            len(train_dataloader.dataset),
                                            round(100. * batch_idx / len(train_dataloader), 2),
                                            round(sum(loss_log)/len(loss_log), 6)))
            print("Train Epoch: {} [{}/{} ({}%)]\t Loss: {}".format(epoch,
                                            (batch_idx + 1) * input_ids.shape[0],
                                            len(train_dataloader.dataset),
                                            round(100. * batch_idx / len(train_dataloader), 2),
                                            round(sum(loss_log)/len(loss_log), 6)))
            f.close()
            loss_log = []
    return global_step
    
def valid_step(model, valid_dataloader, device, log_name
               ):
    model.eval()
    loss_log = []
    for batch_idx, [input_ids, userid, itemid, rating, curr_flag,
                    rating_inputs
                   ] in enumerate(valid_dataloader):
        input_ids = input_ids.to(device)
        itemid = itemid.to(device)
        userid = userid.to(device)
        rating = rating.to(device)
        curr_flag = curr_flag.to(device)
        rating_inputs = rating_inputs.to(device)
        with torch.no_grad():
            with autocast():
                loss = model.train_step(input_ids,  userid, itemid, rating, curr_flag,
                                        rating_inputs )
        loss_log.append(loss.item())

    f = open(log_name,'a+')
    f.write("valid Loss: {}\n".format(round(sum(loss_log)/len(loss_log), 6)))
    print("valid Loss: {}".format( round(sum(loss_log)/len(loss_log), 6)))
    f.close()


    return round(sum(loss_log)/len(loss_log), 6)

def test_step(model, test_dataloader, device, log_name,
            dataset, output_dir, word, tokenizer):
    model.eval()
    prepare_model_for_generation(model)
    predict = []
    label = []
    lens = len(test_dataloader)
    test_pred = []
    test_true = []
    for batch_idx, [input_ids, userid, itemid, rating, curr_flag,
                    rating_inputs
                   ] in enumerate(test_dataloader):
        print('\r',batch_idx,'/',lens,end='')
        input_ids = input_ids.to(device)
        itemid = itemid.to(device)
        userid = userid.to(device)
        rating = rating.to(device)
        curr_flag = curr_flag.to(device)
        
        rating_inputs = rating_inputs.to(device)
        
        text =  torch.tensor([[]]).to(device)
        last_words = torch.tensor([[]]).to(device)
        kv_cache = None
        for idx in range(word):
            with torch.no_grad():
                with autocast():
                    if idx == 0:
                        pre_rating = model.rating_predict(userid, itemid)
                        batch_true = rating.cpu()
                        batch_pred = pre_rating.detach().cpu().numpy()
                        for item in batch_pred:
                            test_pred.append((item*[1.,2.,3.,4.,5.]).sum().item())
                        for item in np.array(batch_true):
                            test_true.append(item+1)
                    logits, kv_cache = model(last_words, userid, itemid, pre_rating, kv_cache)
                    
            word_prob = logits.exp()
            last_words = torch.argmax(word_prob, dim=1).unsqueeze(1)
            if text.shape[1]==0:
                text = last_words
            else:
                text = torch.cat([text, last_words], 1)  
        predict.extend(text.tolist())
        label.extend(input_ids.tolist())
        
    tokens_predict = [ids2words(ids_clear(ids), tokenizer) for ids in predict]
    predict_text = []
    for row in tqdm(predict):
        temp = []
        for item in row:
            if item == 2:
                break
            temp.append(item)
        predict_text.append(temp)
    result = pd.DataFrame({"text":predict_text,"rating":test_pred})
    result.to_pickle(output_dir)    
        
    
    f = open(log_name,'a+')
    # rating
    predicted_rating = [(r, p) for (r, p) in zip(test_true, test_pred)]
    RMSE = root_mean_square_error(predicted_rating, 5, 1)
    f.write('RMSE {:7.4f}\n'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, 5, 1)
    f.write('MAE {:7.4f}\n'.format(MAE))
    # text
    tokens_test = [ids2words(ids_clear(ids), tokenizer) for ids in label]
    tokens_predict = [ids2words(ids_clear(ids), tokenizer) for ids in predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    f.write('BLEU-1 {:7.4f}\n'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    f.write('BLEU-4 {:7.4f}\n'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    f.write('USR {:7.4f} | USN {:7}\n'.format(USR, USN))
    
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    
    feature_set = dataset.feature_set
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    f.write('DIV {:7.4f}\n'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    f.write('FCR {:7.4f}\n'.format(FCR))
    FMR = feature_matching_ratio(feature_batch, dataset.features)
    f.write('FMR {:7.4f}\n'.format(FMR))
    
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    
    ROUGE = rouge_score(text_test, text_predict) 
    for (k, v) in ROUGE.items():
        f.write('{} {:7.4f}\n'.format(k, v))
    f.close()
    return 
    
    
    
def main(args):
    if args.lora_type == 'dora' and 'use_dora' not in signature(LoraConfig).parameters:
        raise RuntimeError('DoRA requires peft>=0.9.0; upgrade PEFT before using --lora_type dora.')
    if args.accumulation_steps <= 0:
        raise ValueError('--accumulation_steps must be positive.')
    for root_dir in (args.ckpt_dir, args.output_dir, args.log_dir):
        os.makedirs(os.path.join(root_dir, args.dataset_name), exist_ok=True)
    seed_everything(args.seed)
    device = 0
    # device = 'cpu'
    if 'MoviesAndTV' in args.dataset_name:
        user_num = 7506
        item_num = 7360
    elif 'ClothingShoesAndJewelry' in args.dataset_name:
        user_num = 38764 
        item_num = 22919
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model_tag = args.model_name.split('/')[-2]
    cache_path = args.data_dir + args.dataset_name + f'/dataset_keywords_{model_tag}.pickle'
    if os.path.exists(cache_path) is False:
        dataset = pd.read_pickle(args.data_dir+args.dataset_name+'/reviews.pickle')
        dataset = pd.DataFrame(dataset)
        itemid = np.array(dataset['item'].tolist())
        userid = np.array(dataset['user'].tolist())
        encoder = LabelEncoder()  
        userid = encoder.fit_transform(userid).tolist()
        itemid = encoder.fit_transform(itemid).tolist()
        dataset['user'] = userid
        dataset['item'] = itemid

        keywords, keywords_words, text = [], [], []
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        bos_id = tokenizer.bos_token_id
        
        for row in tqdm(dataset['template']):
            kw_tokens = tokenizer(row[0])['input_ids']
            # Only when the first token is indeed 'bos' should it be removed.
            if bos_id is not None and len(kw_tokens) > 0 and kw_tokens[0] == bos_id:
                kw_tokens = kw_tokens[1:]
                
            txt_tokens = tokenizer(row[2])['input_ids']
            if bos_id is not None and len(txt_tokens) > 0 and txt_tokens[0] == bos_id:
                txt_tokens = txt_tokens[1:]
                
            keywords.append(kw_tokens)
            keywords_words.append(row[0])
            text.append(txt_tokens + [eos_id])
        dataset['text'] = text
        dataset['keyword'] = keywords
        dataset['keyword_words']=keywords_words
        dataset = dataset[['user','item','text','keyword','keyword_words','rating']]
        dataset.to_pickle(cache_path)
    else:
        dataset = pd.read_pickle(cache_path)
    dataset['rating'] = [int(x-1) for x in dataset['rating'].tolist()]
    
    module_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules = module_list[:args.lora_modules]  # Set as original paper: ["q_proj", "k_proj"]
    image_embeddings = build_multimodal_image_embeddings(args, item_num, device)
    if args.prepare_image_embeddings_only:
        print("Image embedding cache is ready. Stop before loading the LLM.")
        return
    for split_index in parse_split_indices(args.split_indices):
        print(f"========== split_index: {split_index} ==========")
        checkpoint_dir = os.path.join(args.ckpt_dir, args.dataset_name)
        adapter_ckpt_path = os.path.join(checkpoint_dir, f'{split_index}{args.lora_type}_model.pth')
        peft_prefix = (f'{split_index}{args.lora_type}_'
                       if args.lora_type in ('dora', 'adalora') else split_index)
        peft_adapter_dir = os.path.join(checkpoint_dir, f'{peft_prefix}model')
        prompt_encoder_path = os.path.join(checkpoint_dir, f'{peft_prefix}ped.bin')
        prompt_image_path = os.path.join(checkpoint_dir, f'{peft_prefix}prompt_img.bin')
    #for split_index in ['1']:
        train_dataset, valid_dataset, test_dataset = dataset_split(dataset,split_index,args)
        train_set = MyDataset(train_dataset)
        valid_set = MyDataset(valid_dataset)
        test_set = MyDataset(test_dataset)
        collate_train = MyCollater(args.epochs*len(train_dataset)//args.batch_size,args.word,args.delta)
        collate_valid = MyCollater(1,args.word)
        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_train, shuffle=True, pin_memory=True, num_workers=1)
        valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate_valid, shuffle=False)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_valid, shuffle=False)

        total_step = args.epochs * ((len(train_dataloader) + args.accumulation_steps - 1)
                                    // args.accumulation_steps)
        if args.lora_type in ('adalora', 'modadalora') and not args.only_eval:
            if not 0 < args.adalora_target_r < args.adalora_init_r:
                raise ValueError('AdaLoRA requires 0 < --adalora_target_r < --adalora_init_r.')
            if args.adalora_tinit < 0 or args.adalora_tfinal < 0 or args.adalora_tinit + args.adalora_tfinal >= total_step:
                raise ValueError('AdaLoRA requires total optimizer steps > tinit + tfinal.')
            if args.adalora_delta_t <= 0 or args.adalora_orth_reg_weight < 0:
                raise ValueError('AdaLoRA delta_t must be positive and orth_reg_weight nonnegative.')

        model_llm = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map='cuda:'+str(device)
        )
        configure_gradient_checkpointing(model_llm, args.lora_type)

        if args.lora_type in ('modlora', 'moddora', 'modadalora'):
            model_class = {'modlora': MoDLoRA, 'moddora': MoDDoRA,
                           'modadalora': MoDAdaLoRA}[args.lora_type]
            model = model_class(user_num=user_num, 
                             item_num=item_num, 
                             hidden=args.id_hidden, 
                             llm_hidden=model_llm.config.hidden_size, 
                             tokenizer=tokenizer, 
                             model_llm=model_llm, 
                             r=args.adalora_init_r if args.lora_type == 'modadalora' else args.r,
                             lora_modules=args.lora_modules,
                             image_embeddings=image_embeddings,
                             ui_multimodal_scale=args.ui_multimodal_scale,
                             image_multimodal_scale=args.image_multimodal_scale).to(device)
        else:
            # Original Method，using PEFT
            lora_kwargs = dict(
                r=args.r,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            if args.lora_type == 'dora':
                lora_kwargs['use_dora'] = True
            if args.lora_type == 'adalora':
                lora_kwargs.pop('r')
                lora_config = AdaLoraConfig(
                    **lora_kwargs,
                    init_r=args.adalora_init_r,
                    target_r=args.adalora_target_r,
                    total_step=total_step,
                    tinit=args.adalora_tinit,
                    tfinal=args.adalora_tfinal,
                    deltaT=args.adalora_delta_t,
                    orth_reg_weight=args.adalora_orth_reg_weight,
                )
            else:
                lora_config = LoraConfig(**lora_kwargs)
            model_llm = get_peft_model(model_llm, lora_config)
            model_llm.print_trainable_parameters()
            model = MyModel(
                user_num,
                item_num,
                args.id_hidden,
                model_llm.config.hidden_size,
                tokenizer,
                image_embeddings=image_embeddings
            ).to(device)
            model.model = model_llm
        
        model.generate_weight = args.generate_weight
        model.rating_weight = args.rating_weight

        param = list(filter(lambda p: p.requires_grad==True, model.prompt_encoder.parameters()))
        if getattr(model, "f_prompt_img", None) is not None:
            param += list(filter(lambda p: p.requires_grad==True, model.f_prompt_img.parameters()))
        if args.lora_type in ('modlora', 'moddora', 'modadalora'):
            param += list(filter(lambda p: p.requires_grad==True, model.f_ui.parameters()))
            if getattr(model, "f_img", None) is not None:
                param += list(filter(lambda p: p.requires_grad==True, model.f_img.parameters()))
            
        param2 = filter(lambda p: p.requires_grad==True, model.model.parameters())
        optimizer = AdamW([
             {'params': param, 'lr': args.learning_rate},
             {'params': param2, 'lr': args.learning_rate/10},
        ])
        rank_update = None
        if args.lora_type == 'adalora':
            rank_update = model.model.base_model.update_and_allocate
        elif args.lora_type == 'modadalora' and not args.only_eval:
            rank_allocator = MoDAdaLoRARankAllocator(
                model, args.adalora_init_r, args.adalora_target_r, total_step,
                args.adalora_tinit, args.adalora_tfinal, args.adalora_delta_t,
            )
            rank_update = rank_allocator.update_and_allocate

        log_name = os.path.join(args.log_dir, args.dataset_name, f'{args.lora_type}_{args.log_name}')
        output_dir = os.path.join(args.output_dir, args.dataset_name,
                                  f'{split_index}{args.lora_type}_generate.dataset')
        f = open(log_name,'a+')
        f.write(args.model_name+"\n")
        f.write("                                 split_index:" +split_index+"\n")
        f.close()
        log_trainable_parameters_summary(model, log_name)
        best_loss = 999
        early_stop = 1
        reset_peak_vram_stats(device)

        global_step = 0
        if args.only_eval == False:
            for epoch in range(0, args.epochs):
                global_step = train_step(model, train_dataloader,
                               optimizer, device,
                               epoch, args.show_train_loss_steps,
                               args.accumulation_steps,
                               log_name,
                               global_step=global_step,
                               rank_update=rank_update,
                               orth_reg_weight=(args.adalora_orth_reg_weight
                                               if args.lora_type in ('adalora', 'modadalora') else 0.0),
                        )
                collate_train.cur_step = len(train_dataloader)*(epoch+1)
                valid_loss = valid_step(model, valid_dataloader, device, log_name)
    
                f = open(log_name,'a+')
                print(valid_loss)
                f.write(str(valid_loss)+'\n')
                if (args.lora_type in ('adalora', 'modadalora')
                        and global_step < total_step - args.adalora_tfinal):
                    f.write('Waiting for the final AdaLoRA rank budget before checkpoint selection.\n')
                    f.close()
                    continue
                if best_loss<valid_loss:
                    early_stop -= 1
                else:
                    print("save model\n")
                    f.write("save model\n")
                    best_loss = valid_loss
                    
                    if args.lora_type in ('modlora', 'moddora', 'modadalora'):
                        trainable_params = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
                        torch.save(trainable_params, adapter_ckpt_path)
                    else:
                        model.model.save_pretrained(peft_adapter_dir)
                        torch.save(model.prompt_encoder, prompt_encoder_path)
                        if getattr(model, "f_prompt_img", None) is not None:
                            torch.save(
                                model.f_prompt_img.state_dict(),
                                prompt_image_path
                            )
                f.close()
                if early_stop == 0:
                    break

        if args.lora_type in ('modlora', 'moddora', 'modadalora'):
            torch.cuda.empty_cache()
            model.load_state_dict(load_adapter_state_dict(adapter_ckpt_path, map_location="cpu"), strict=False)
        else:
            adapter_name = f'best_{args.lora_type}'
            model.model.load_adapter(peft_adapter_dir, adapter_name)
            model.model.set_adapter(adapter_name)
            model.prompt_encoder = torch.load(prompt_encoder_path, map_location="cuda:"+str(device), weights_only=False)
            if getattr(model, "f_prompt_img", None) is not None:
                if os.path.exists(prompt_image_path):
                    model.f_prompt_img.load_state_dict(load_torch_object(prompt_image_path, map_location="cpu"))
                else:
                    print(f"Warning: prompt image projector checkpoint not found: {prompt_image_path}")
            
        test_step(model, test_dataloader, device, log_name, test_set, output_dir, args.word, tokenizer)
        log_vram_stats(device, log_name, include_peak=True)

        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_vram_stats(device, log_name)

        del model, model_llm
        del train_dataloader, valid_dataloader, test_dataloader
        del train_set, valid_set, test_set
        del train_dataset, valid_dataset, test_dataset
        cleanup_split()

   
        
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--devices', default=-1, type=int,
                       help='Select which GPU to use with the program.')

    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--seed', default=5254, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument('--rating_weight', default=0.1, type=float,
                       help='regularization on recommendation task')
    parser.add_argument('--generate_weight', default=1.0, type=float,
                       help='regularization on generation task')
    parser.add_argument('--delta', default=0.2, type=float)
    parser.add_argument('--word', default=20, type=int,
                       help='number of words to generate for each sample')
    parser.add_argument('--show_train_loss_steps', default=500, type=int,
                       help='number of train steps for display the loss')
    parser.add_argument('--id_hidden', default=1024, type=int)
    parser.add_argument('--only_eval', action='store_true')

    parser.add_argument('--dataset_name', default='MoviesAndTV', type=str)
    parser.add_argument('--split_indices', default='1,2,3,4,5', type=str,
                       help='comma-separated split indices to run, e.g. 1 or 1,2,3,4,5')
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--model_name', default='../autodl-fs/Qwen2.5-7B/', type=str)
    parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    parser.add_argument('--log_dir', default='./log/', type=str)
    parser.add_argument('--log_name', default='llama.log', type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)
    parser.add_argument('--lora_type', choices=['lora', 'modlora', 'dora', 'moddora', 'adalora', 'modadalora'], default='lora',
                        help='fine-tuning method: PEFT LoRA/DoRA/AdaLoRA or custom MoD variants')
    parser.add_argument('--lora_modules', '--lora_modules', type=int, default=2, help='number of modules for LoRA')
    parser.add_argument('--r', '--r', type=int, default=4, help='rank for LoRA')
    parser.add_argument('--adalora_init_r', default=12, type=int, help='initial rank per AdaLoRA branch')
    parser.add_argument('--adalora_target_r', default=4, type=int, help='final average rank per AdaLoRA branch')
    parser.add_argument('--adalora_tinit', default=0, type=int, help='AdaLoRA warmup optimizer steps')
    parser.add_argument('--adalora_tfinal', default=0, type=int, help='AdaLoRA final fixed-rank optimizer steps')
    parser.add_argument('--adalora_delta_t', default=1, type=int, help='AdaLoRA rank allocation interval')
    parser.add_argument('--adalora_orth_reg_weight', default=0.5, type=float,
                        help='AdaLoRA factor orthogonality regularization weight')
    parser.add_argument('--use_multimodal', action='store_true', help='Enable item-image features for CIER and MoD adapters')
    parser.add_argument('--clip_model', default=None, type=str, help='CLIP model path for generating item image embeddings')
    parser.add_argument('--image_dir', default='images', type=str, help='image folder name under the dataset directory')
    parser.add_argument('--image_embedding_path', default=None, type=str, help='optional precomputed image embedding file')
    parser.add_argument('--prepare_image_embeddings_only', action='store_true', help='only build/load image embedding cache and exit')
    parser.add_argument('--ui_multimodal_scale', default=2.0, type=float, help='initial trainable scaling value for user-item LoRA branch')
    parser.add_argument('--image_multimodal_scale', default=2.0, type=float, help='initial trainable scaling value for image LoRA branch')

    args = parser.parse_args()

    main(args)
