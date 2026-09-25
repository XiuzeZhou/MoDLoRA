import os
import math
import torch
import argparse
from torch.optim import AdamW
from transformers import AutoTokenizer
from module import MoDLoRA, Concat_LoRA  
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


parser = argparse.ArgumentParser(description='User-Item Adapter for LLM-based Explainable Recommendation Systems (MoDLoRA)')
parser.add_argument('-data_path', '--data_path', type=str, default='./data/ClothingShoesAndJewelry/reviews.pickle',
                    help='path for loading the pickle data')
parser.add_argument('-index_dir', '--index_dir', type=str, default='./data/ClothingShoesAndJewelry/1/',
                    help='load indexes')
parser.add_argument('-llm_model', '--llm_model', type=str, default="./llm/Qwen2.5-7B",
                    help='LLM backbone')
parser.add_argument('-clip_model', '--clip_model', type=str, default="./llm/clip-vit-base-patch32/",
                    help='CLIP model path used to generate item image embeddings')
parser.add_argument('-model_type', '--model_type', type=str, default="lora", choices=['modlora', 'lora'],
                    help='model architecture pipeline: multimodal ui/image LoRA or multimodal concat')
parser.add_argument('-lr', '--lr', type=float, default=1e-7,
                    help='learning rate for the model')
parser.add_argument('-epochs', '--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('-batch_size', '--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('-cuda', '--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('-log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('-checkpoint', '--checkpoint', type=str, default='./checkpoint/',
                    help='directory to save the final model')
parser.add_argument('-outf', '--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('-endure_times', '--endure_times', type=int, default=1,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('-words', '--words', type=int, default=20,
                    help='number of words to generate for each sample')
parser.add_argument('-rating_reg', '--rating_reg', type=float, default=0.1, 
                    help='regularization on rating task')
parser.add_argument('-mlp_size', '--mlp_size', type=int, default=300, 
                    help='hidden size of MLP')
parser.add_argument('-k', '--k', type=int, default=64, 
                    help='dimensional size of user/item embeddings')
parser.add_argument('-r', '--r', type=int, default=8, 
                    help='rank for Textual LoRA')
parser.add_argument('-lora_modules', '--lora_modules', type=int, default=7, 
                    help='number of modules for LoRA')
parser.add_argument('-image_dir', '--image_dir', type=str, default='images',
                    help='image folder name under the dataset directory')
parser.add_argument('-ui_multimodal_scale', '--ui_multimodal_scale', type=float, default=2.0,
                    help='initial trainable scaling value for the user-item LoRA branch')
parser.add_argument('-image_multimodal_scale', '--image_multimodal_scale', type=float, default=2.0,
                    help='initial trainable scaling value for the image LoRA branch')
parser.add_argument('-clip_norm', '--clip_norm', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('-acc_steps', '--acc_steps', type=int, default=64,
                    help='steps of gradient accumulation')
parser.add_argument('-seed', '--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')
torch.cuda.reset_peak_memory_stats(device)

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = args.outf

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side='left', spaces_between_special_tokens=False) 

if 'llama' in args.llm_model.lower() or 'qwen' in args.llm_model.lower() or 'mistral' in args.llm_model.lower() or 'gemma' in args.llm_model.lower():  
    bos = tokenizer.bos_token or '<s>'
    eos = tokenizer.eos_token or '</s>'
    pad = tokenizer.pad_token or tokenizer.eos_token

tokenizer.pad_token = pad  # Ensure pad_token is set consistently
tokenizer.add_special_tokens({'bos_token': bos, 'eos_token': eos, 'pad_token': pad})
corpus = DataLoader(
    args.data_path, args.index_dir, tokenizer, args.words,
    clip_path=args.clip_model if args.model_type in ['modlora', 'lora'] else None,
    image_dir=args.image_dir,
    device=device
)
feature_set = corpus.feature_set

train_data = Batchify(corpus.train, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, args.words, args.batch_size, corpus.max_rating, corpus.min_rating, shuffle=True)
val_data = Batchify(corpus.valid, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, args.words, args.batch_size, corpus.max_rating, corpus.min_rating)
test_data = Batchify(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, args.words, args.batch_size*4, corpus.max_rating, corpus.min_rating)

###############################################################################
# Build the model
###############################################################################

if args.model_type == 'modlora':
    model = MoDLoRA.from_pretrained(
        args.llm_model, len(corpus.user_dict), len(corpus.item_dict),
        args.k, args.r, args.mlp_size, args.lora_modules,
        image_embeddings=corpus.image_embeddings,
        ui_multimodal_scale=args.ui_multimodal_scale,
        image_multimodal_scale=args.image_multimodal_scale
    )
else:
    model = Concat_LoRA.from_pretrained(
        args.llm_model, len(corpus.user_dict), len(corpus.item_dict),
        args.k, args.r, args.mlp_size, args.lora_modules,
        image_embeddings=corpus.image_embeddings
    )

model.resize_token_embeddings(len(tokenizer))
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################
import torch.nn.functional as F

def train(train_data):
    model.train()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    accum_count = 0
    while True:
        user, item, rating, input_ids, mask, prompt_ids, prompt_text, prompt_lens, review_text, text_lens = train_data.next_batch() 
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)
        input_ids = input_ids.to(device)
        mask = mask.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs, hat_r = model(input_ids=input_ids, attention_mask=mask, user=user, item=item, text_lens=text_lens)
            L_r = F.mse_loss(hat_r.to(torch.float32), rating.to(torch.float32))
            loss = outputs.loss + args.rating_reg * L_r

            loss = loss / args.acc_steps

        loss.backward()

        accum_count += 1
        batch_size = input_ids.size(0)
        text_loss += batch_size * outputs.loss.item()
        rating_loss += batch_size * L_r.item()
        total_sample += batch_size

        if accum_count % args.acc_steps == 0 or train_data.step == train_data.total_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
        
        if train_data.step % args.log_interval == 0 or train_data.step == train_data.total_step:
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_t_loss), cur_r_loss, train_data.step, train_data.total_step))
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0
        if train_data.step == train_data.total_step:
            break

def evaluate(data):
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, input_ids, mask, prompt_ids, prompt_text, prompt_lens, review_text, text_lens = data.next_batch() 
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            outputs, *_ = model(input_ids=input_ids, attention_mask=mask, user=user, item=item, text_lens=text_lens)
            loss = outputs.loss
            batch_size = input_ids.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size
            if data.step == data.total_step:
                break
    return text_loss / total_sample

def generate(data):
    model.eval()
    idss_predict = []
    input_features = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, input_ids, mask, prompt_ids, prompt_text, prompt_lens, review_text, review_lens = data.next_batch()
            user = user.to(device)
            item = item.to(device)
            batch_size = input_ids.size(0)

            max_p_len = max([len(p) for p in prompt_ids])
            gen_input_ids = torch.full((batch_size, max_p_len + 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(gen_input_ids)
            
            for b_idx in range(batch_size):
                p_list = prompt_ids[b_idx]
                p_len = len(p_list)
                gen_input_ids[b_idx, -1-p_len : -1] = torch.tensor(p_list, dtype=torch.long, device=device)
                gen_input_ids[b_idx, -1] = tokenizer.bos_token_id
                attention_mask[b_idx, -1-p_len:] = 1

            if args.model_type == 'modlora':
                input_size = gen_input_ids.size(1)

            generated_output = model.generate(
                input_ids=gen_input_ids, 
                max_new_tokens=args.words,
                user=user, 
                item=item,
                do_sample=False,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=False,
            )  # Greedy decoding; adjust for beam search if needed
            
            generated_ids = generated_output.sequences
            if args.model_type == 'modlora':
                ids = generated_ids[:, input_size:].tolist()
            else:
                # Concat_LoRA injects [u, i, x_img] through inputs_embeds, so those virtual tokens
                # are not present in generated_output.sequences.
                ids = generated_ids[:, 1:].tolist()
            idss_predict.extend(ids)

            # --- Rating Prediction ---
            # Set multimodal context again as it might be needed for the forward pass
            if args.model_type == 'modlora':
                x_ui, x_img = model.build_lora_context(user, item)
                model.set_Q(x_ui, x_img)

            hat_r = model.predict_rating(user, item)
            rating_predict.extend(hat_r.tolist())

            for p_list in prompt_ids:
                input_features.append(p_list)

            if data.step == data.total_step:
                break
    return idss_predict, input_features, rating_predict


print('='*80)
print(now_time() + f'Tuning LLM with LoRA (Architecture: {args.model_type.upper()})')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trainable_state_dict = {
            k: v.cpu() for k, v in model.named_parameters() if v.requires_grad
        }
        torch.save(trainable_state_dict, model_path)
        print(now_time() + f'Trainable parameters saved to {model_path}')
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

###############################################################################
# Train rating predictor
###############################################################################
from rating_prediction import train_rating_predictor

#model.load_state_dict(torch.load(model_path, map_location=device))
train_rating_predictor(model, train_data, test_data, val_data, model_path, lr=1e-4, max_rating=corpus.max_rating, min_rating=corpus.min_rating, device=device)
if args.model_type in ['lora', 'modlora']:
    train_rating_predictor(model, train_data, test_data, val_data, model_path, lr=1e-4, max_rating=corpus.max_rating, min_rating=corpus.min_rating, device=device)
else:
    print(now_time() + " CIER uses LLM for rating prediction. Skipping separate rating predictor training.")

###############################################################################
# Evaluate
###############################################################################

if torch.cuda.is_available():
    peak_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    
    print("=" * 80)
    print(f"📊 [VRAM Profiling] Peak GPU Memory Allocated: {peak_memory_allocated:.2f} GB")
    print(f"📊 [VRAM Profiling] Peak GPU Memory Reserved (Cached): {peak_memory_reserved:.2f} GB")
    print("=" * 80)

del optimizer
torch.cuda.empty_cache()

print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Load the best saved model.
if args.model_type == 'modlora':
    model = MoDLoRA.from_pretrained(
        args.llm_model, len(corpus.user_dict), len(corpus.item_dict),
        args.k, args.r, args.mlp_size, args.lora_modules,
        image_embeddings=corpus.image_embeddings,
        ui_multimodal_scale=args.ui_multimodal_scale,
        image_multimodal_scale=args.image_multimodal_scale
    )
else:
    model = Concat_LoRA.from_pretrained(
        args.llm_model, len(corpus.user_dict), len(corpus.item_dict),
        args.k, args.r, args.mlp_size, args.lora_modules,
        image_embeddings=corpus.image_embeddings
    )

model.resize_token_embeddings(len(tokenizer))
loaded_state = torch.load(model_path, map_location=device)
model.load_state_dict(loaded_state, strict=False)
print(now_time() + f'Successfully loaded trainable parameters from {model_path}')
model = model.to(device)

import os
rating_model_path = model_path + "_rating.pt"
if os.path.exists(rating_model_path):
    rating_state = torch.load(rating_model_path, map_location=device)
    model.load_state_dict(rating_state, strict=False)
    print(now_time() + f'Successfully loaded Rating Predictor weights from {rating_model_path}')

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} on test | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted, features, rating_predicted = generate(test_data)
# rating
predicted_rating = [(r*corpus.max_rating, p*corpus.max_rating) for (r, p) in zip(test_data.rating.tolist(), rating_predicted)]
RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'MAE {:7.4f}'.format(MAE))

tokens_test = [ids2tokens(ids[-t_len:].tolist(), tokenizer, eos) for ids, t_len in zip(test_data.input_ids, test_data.text_lens)]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))

text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
tokens_context = [tokenizer.decode(ids, skip_special_tokens=True) for ids in features]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''
for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
    text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
