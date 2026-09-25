import os
import re
import math
import torch
import random
import pickle
import datetime
import json
from rouge import rouge
from bleu import compute_bleu


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:
    def __init__(self, data_path, index_dir, tokenizer, seq_len,
                 clip_path=None, image_dir='images',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.user_counts = {}
        self.item_counts = {}
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.image_dim = None
        self._item_image_features = {}
        self.device = device
        self.image_dir = os.path.join(os.path.dirname(data_path), image_dir)
        self.item_json_path = os.path.join(os.path.dirname(data_path), 'item.json')
        self.item_descriptions = {}
        self.item_id_to_asin = {}
        self.initialize(data_path)
        if clip_path:
            self.item_id_to_asin = self.create_item_id_to_asin_mapping(data_path)
            self.generate_item_embeddings(clip_path)
        self.image_embeddings = self.build_image_embeddings()
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)

    def to_image_tensor(self, value):
        if value is None or isinstance(value, str):
            return None
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            return None
        if tensor.numel() == 0:
            return None
        while tensor.dim() > 1:
            tensor = tensor.mean(dim=0)
        return tensor.contiguous()

    def load_item_descriptions(self):
        if not os.path.exists(self.item_json_path):
            print(now_time() + f"Warning: item.json not found at {self.item_json_path}")
            return {}

        with open(self.item_json_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        descriptions = {}
        for item in items:
            if 'item' in item and 'description' in item:
                descriptions[item['item']] = item['description']
        return descriptions

    def create_item_id_to_asin_mapping(self, data_path):
        item_id_to_asin = {}
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            item_idx = self.item_dict.entity2idx[review['item']]
            item_id_to_asin[item_idx] = review['item']
        return item_id_to_asin

    def add_image_embeddings_from_object(self, obj):
        if isinstance(obj, dict):
            for table_key in ['image_embeddings', 'image_embedding', 'item_embeddings', 'embeddings', 'features']:
                if table_key in obj and not isinstance(obj[table_key], (int, float, str)):
                    obj = obj[table_key]
                    break

        if isinstance(obj, dict):
            for item_key, feature in obj.items():
                item_idx = None
                if item_key in self.item_dict.entity2idx:
                    item_idx = self.item_dict.entity2idx[item_key]
                elif isinstance(item_key, int) and 0 <= item_key < len(self.item_dict):
                    item_idx = item_key
                if item_idx is None:
                    continue
                tensor = self.to_image_tensor(feature)
                if tensor is not None:
                    self._item_image_features[item_idx] = tensor
            return

        tensor = torch.as_tensor(obj, dtype=torch.float32)
        if tensor.dim() != 2:
            raise ValueError("Image embeddings should be a 2-D table or a dict keyed by item id")
        item_count = min(len(self.item_dict), tensor.size(0))
        for item_idx in range(item_count):
            self._item_image_features[item_idx] = tensor[item_idx].contiguous()

    def generate_item_embeddings(self, clip_path="./llm/clip-vit-base-patch32/"):
        if not self.item_id_to_asin:
            return

        cache_dir = os.path.join(os.path.dirname(self.image_dir), 'embeddings_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'item_embeddings.pt')

        if os.path.exists(cache_path):
            print(now_time() + f"Loading cached item embedding from {cache_path}")
            try:
                item_embeddings = torch.load(cache_path, map_location='cpu')
                self.add_image_embeddings_from_object(item_embeddings)
                if len(self._item_image_features) >= len(self.item_dict):
                    return
            except Exception as e:
                print(now_time() + f"Error occured in loading cached item embedding: {e}")

        try:
            from PIL import Image
            from tqdm import tqdm
            from transformers import CLIPProcessor, CLIPModel
        except ImportError as e:
            print(now_time() + f"Warning: cannot import CLIP image dependencies: {e}")
            return

        print(now_time() + "Initializing CLIP model...")
        clip_model = CLIPModel.from_pretrained(clip_path).to(self.device)
        clip_processor = CLIPProcessor.from_pretrained(clip_path)
        self.image_dim = getattr(clip_model.config, "projection_dim", None)
        for param in clip_model.parameters():
            param.requires_grad = False

        item_embeddings = {}
        print(now_time() + "Generating item embeddings with CLIP...")
        print(now_time() + f"Detected {len(self.item_id_to_asin)} items")

        clip_model.to(self.device)
        self.item_descriptions = self.load_item_descriptions()
        for item_idx, asin in tqdm(self.item_id_to_asin.items()):
            image_path = os.path.join(self.image_dir, f"{asin}.jpg")
            image_embedding = None

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_inputs = clip_processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        image_embedding = clip_model.get_image_features(**image_inputs).cpu()
                except Exception as e:
                    print(now_time() + f"Error processing image {asin}: {e}")
            else:
                white_image = Image.new("RGB", (300, 300), (255, 255, 255))
                white_image.info['dpi'] = (96, 96)
                image_inputs = clip_processor(images=white_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_embedding = clip_model.get_image_features(**image_inputs).cpu()

            if image_embedding is None and asin in self.item_descriptions:
                description = self.item_descriptions[asin]
                try:
                    text_inputs = clip_processor(text=[description], return_tensors="pt", padding=True, truncation=True).to(self.device)
                    with torch.no_grad():
                        image_embedding = clip_model.get_text_features(**text_inputs).cpu()
                except Exception as e:
                    print(now_time() + f"Error processing description for {asin}: {e}")

            tensor = self.to_image_tensor(image_embedding)
            if tensor is not None:
                item_embeddings[item_idx] = image_embedding
                self._item_image_features[item_idx] = tensor

        print(now_time() + f"Generated embeddings for {len(item_embeddings)} items")

        try:
            torch.save(item_embeddings, cache_path)
            print(now_time() + f"Item embedding saved to {cache_path}")
        except Exception as e:
            print(now_time() + f"Error occured when saving item embedding: {e}")

        print(now_time() + "Deleting CLIP model from the project to avoid unnecessary performance loss")
        del clip_model
        del clip_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_image_embeddings(self):
        if not self._item_image_features:
            return None

        first_feature = next(iter(self._item_image_features.values()))
        feature_dim = int(self.image_dim) if self.image_dim is not None else first_feature.numel()
        image_embeddings = torch.zeros((len(self.item_dict), feature_dim), dtype=torch.float32)
        for item_idx, feature in self._item_image_features.items():
            feature = feature.flatten().to(torch.float32)
            copy_dim = min(feature_dim, feature.numel())
            image_embeddings[item_idx, :copy_dim] = feature[:copy_dim]

        self.image_dim = feature_dim
        return image_embeddings

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating
            u_id = review['user']
            i_id = review['item']
            self.user_counts[u_id] = self.user_counts.get(u_id, 0) + 1
            self.item_counts[i_id] = self.item_counts.get(i_id, 0) + 1

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            tokens = self.tokenizer(tem)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len], skip_special_tokens=True)  # keep seq_len tokens at most
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': text,
                         'feature': fea})
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index


class Batchify:
    def __init__(self, data, user2feature, item2feature, tokenizer, bos, eos, seq_len, batch_size=128, max_rating=5.0, min_rating=1.0, shuffle=False):
        u, i, r = [], [], []
        self.feature = []
        self.prompt_text = []
        self.review_text = []
        self.tokenizer = tokenizer
        
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'] / max_rating)
            
            ufea = set(user2feature[x['user']])
            ifea = set(item2feature[x['item']])
            intersection = ufea & ifea
            difference = (ufea | ifea) - intersection
            feature_list = list(intersection) + list(difference)
            f = ' '.join(feature_list)
            
            tokens = tokenizer(f)['input_ids']
            text = tokenizer.decode(tokens[:seq_len], skip_special_tokens=True) 
            
            self.prompt_text.append(text)
            self.review_text.append('{} {} {}'.format(bos, x['text'], eos))
            self.feature.append(x['feature'])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()

        # Left-Padding
        prompt_ids_list = [tokenizer(p, add_special_tokens=False)['input_ids'] for p in self.prompt_text]
        review_ids_list = [tokenizer(t, add_special_tokens=False)['input_ids'] for t in self.review_text]
        
        self.prompt_lens = [len(p) for p in prompt_ids_list]
        self.text_lens = [len(t) for t in review_ids_list]
        
        valid_total_lens = [p_len + t_len for p_len, t_len in zip(self.prompt_lens, self.text_lens)]
        max_valid_len = max(valid_total_lens) 
        
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None: pad_id = 0

        num_samples = len(data)
        self.input_ids = torch.full((num_samples, max_valid_len), pad_id, dtype=torch.int64)
        self.attention_mask = torch.zeros((num_samples, max_valid_len), dtype=torch.int64)
        self.raw_prompt_ids_list = prompt_ids_list

        for idx in range(num_samples):
            p_ids = prompt_ids_list[idx]
            r_ids = review_ids_list[idx]
            combined_valid = p_ids + r_ids
            v_len = len(combined_valid)
            
            # Position: [Pad, Pad, ..., Prompt, <bos>, Review, <eos>] ──────────────────
            #                                                          ▲ 绝对右对齐
            start_pos = max_valid_len - v_len
            
            self.input_ids[idx, start_pos:] = torch.tensor(combined_valid, dtype=torch.int64)
            self.attention_mask[idx, start_pos:] = 1

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = num_samples
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]

        user = self.user[index]
        item = self.item[index]
        rating = self.rating[index]

        input_ids = self.input_ids[index]
        mask = self.attention_mask[index]
        
        review_text = [self.review_text[idx] for idx in index]
        text_lens = [self.text_lens[idx] for idx in index]
        prompt_text = [self.prompt_text[idx] for idx in index]
        prompt_lens = [self.prompt_lens[idx] for idx in index]
        
        prompt_ids = [self.raw_prompt_ids_list[idx] for idx in index]

        return user, item, rating, input_ids, mask, prompt_ids, prompt_text, prompt_lens, review_text, text_lens


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens
