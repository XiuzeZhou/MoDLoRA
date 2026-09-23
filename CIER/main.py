import torch
import random
import os
import gc
import torch.nn as nn
import pandas as pd
import numpy as np
import re
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
from peft import LoraConfig, get_peft_model, TaskType
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

def configure_gradient_checkpointing(model_llm, use_modlora):
    model_llm.config.use_cache = False
    if use_modlora:
        try:
            model_llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            print("Non-reentrant gradient checkpointing is not supported; disabling it for MoDLoRA.")
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

def train_step(model, 
               train_dataloader,
               optimizer,
               device,
               epoch, 
               show_train_loss_steps,
               accumulation_steps,
               log_name
               ):
    model.train()
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
        with autocast():
            loss = model.train_step(input_ids, userid, itemid, rating, curr_flag,
                                    rating_inputs
                                   )
        loss_log.append(loss.item())
        loss = loss / accumulation_steps
        
        loss.backward() 
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() 
            model.zero_grad()
        
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
    return    
    
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
    if not os.path.exists(args.ckpt_dir + args.dataset_name):
        os.makedirs(args.ckpt_dir + args.dataset_name)
    if not os.path.exists(args.output_dir + args.dataset_name):
        os.makedirs(args.output_dir + args.dataset_name)
    if not os.path.exists(args.log_dir + args.dataset_name):
        os.makedirs(args.log_dir + args.dataset_name)
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

        # Define LoRA Config
        lora_config = LoraConfig(
         r=args.r,
         lora_alpha=32,
         target_modules=target_modules,
         lora_dropout=0.05,
         bias="none",
         task_type=TaskType.CAUSAL_LM
        )
        model_llm = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map='cuda:'+str(device)
        )
        configure_gradient_checkpointing(model_llm, args.use_modlora)

        if args.use_modlora:
            model = MoDLoRA(user_num=user_num, 
                             item_num=item_num, 
                             hidden=args.id_hidden, 
                             llm_hidden=model_llm.config.hidden_size, 
                             tokenizer=tokenizer, 
                             model_llm=model_llm, 
                             r=args.r, 
                             lora_modules=args.lora_modules,
                             image_embeddings=image_embeddings,
                             ui_multimodal_scale=args.ui_multimodal_scale,
                             image_multimodal_scale=args.image_multimodal_scale).to(device)
        else:
            # Original Method，using PEFT
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
        if args.use_modlora:
            param += list(filter(lambda p: p.requires_grad==True, model.f_ui.parameters()))
            if getattr(model, "f_img", None) is not None:
                param += list(filter(lambda p: p.requires_grad==True, model.f_img.parameters()))
            
        param2 = filter(lambda p: p.requires_grad==True, model.model.parameters())
        optimizer = AdamW([
             {'params': param, 'lr': args.learning_rate},
             {'params': param2, 'lr': args.learning_rate/10},
        ])

        log_name = args.log_dir + args.dataset_name+'/' + args.log_name
        output_dir = args.output_dir+ args.dataset_name+'/'+split_index+'generate.dataset'
        f = open(log_name,'a+')
        f.write(args.model_name+"\n")
        f.write("                                 split_index:" +split_index+"\n")
        f.close()
        best_loss = 999
        early_stop = 1

        if args.only_eval == False:
            for epoch in range(0, args.epochs):
                a = train_step(model, train_dataloader,
                               optimizer, device,
                               epoch, args.show_train_loss_steps,
                               args.accumulation_steps,
                               log_name
                        )
                collate_train.cur_step = len(train_dataloader)*(epoch+1)
                valid_loss = valid_step(model, valid_dataloader, device, log_name)
    
                f = open(log_name,'a+')
                print(valid_loss)
                f.write(str(valid_loss)+'\n')
                if best_loss<valid_loss:
                    early_stop -= 1
                else:
                    print("save model\n")
                    f.write("save model\n")
                    best_loss = valid_loss
                    
                    if args.use_modlora:
                        adapter_ckpt_path = args.ckpt_dir + args.dataset_name + '/' + split_index + 'modlora_model.pth'
                        trainable_params = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
                        torch.save(trainable_params, adapter_ckpt_path)
                    else:
                        model.model.save_pretrained(args.ckpt_dir + args.dataset_name + '/'+split_index+'model')
                        torch.save(model.prompt_encoder,args.ckpt_dir + args.dataset_name + '/'+split_index+'ped.bin')
                        if getattr(model, "f_prompt_img", None) is not None:
                            torch.save(
                                model.f_prompt_img.state_dict(),
                                args.ckpt_dir + args.dataset_name + '/' + split_index + 'prompt_img.bin'
                            )
                f.close()
                if early_stop == 0:
                    break

        if args.use_modlora:
            adapter_ckpt_path = args.ckpt_dir + args.dataset_name + '/' + split_index + 'modlora_model.pth'
            torch.cuda.empty_cache()
            model.load_state_dict(load_adapter_state_dict(adapter_ckpt_path, map_location="cpu"), strict=False)
        else:
            model.model.load_adapter(args.ckpt_dir + args.dataset_name + '/'+split_index+'model', 'best_lora')
            model.model.set_adapter("best_lora")
            model.prompt_encoder = torch.load(args.ckpt_dir + args.dataset_name + '/'+split_index+'ped.bin', map_location="cuda:"+str(device), weights_only=False)
            prompt_img_path = args.ckpt_dir + args.dataset_name + '/' + split_index + 'prompt_img.bin'
            if getattr(model, "f_prompt_img", None) is not None:
                if os.path.exists(prompt_img_path):
                    model.f_prompt_img.load_state_dict(load_torch_object(prompt_img_path, map_location="cpu"))
                else:
                    print(f"Warning: prompt image projector checkpoint not found: {prompt_img_path}")
            
        test_step(model, test_dataloader, device, log_name, test_set, output_dir, args.word, tokenizer)
        del model, model_llm, optimizer
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
    parser.add_argument('--use_modlora', action='store_true', help='Use MoDLoRA architecture')
    parser.add_argument('--lora_modules', '--lora_modules', type=int, default=2, help='number of modules for LoRA')
    parser.add_argument('--r', '--r', type=int, default=4, help='rank for LoRA')
    parser.add_argument('--use_multimodal', action='store_true', help='Enable item-image features for CIER/MoDLoRA')
    parser.add_argument('--clip_model', default=None, type=str, help='CLIP model path for generating item image embeddings')
    parser.add_argument('--image_dir', default='images', type=str, help='image folder name under the dataset directory')
    parser.add_argument('--image_embedding_path', default=None, type=str, help='optional precomputed image embedding file')
    parser.add_argument('--prepare_image_embeddings_only', action='store_true', help='only build/load image embedding cache and exit')
    parser.add_argument('--ui_multimodal_scale', default=2.0, type=float, help='initial trainable scaling value for user-item LoRA branch')
    parser.add_argument('--image_multimodal_scale', default=2.0, type=float, help='initial trainable scaling value for image LoRA branch')

    args = parser.parse_args()

    main(args)
