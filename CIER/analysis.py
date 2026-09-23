import os
import random
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataloader import MyCollater, MyDataset, dataset_split
from model import MultiModalLoraLayer, MoDLoRA


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


def infer_user_item_num(dataset_name):
    if 'Yelp' in dataset_name:
        return 27147, 20266
    if 'TripAdvisor' in dataset_name:
        return 9765, 6280
    if 'MoviesAndTV' in dataset_name:
        return 7506, 7360
    if 'ClothingShoesAndJewelry' in dataset_name:
        return 38764, 22919
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def model_tag_from_path(model_name):
    return os.path.basename(os.path.normpath(model_name))


# Keep dataset preprocessing aligned with main.py; analysis-specific code starts later.
def load_or_build_dataset(args, tokenizer):
    dataset_dir = os.path.join(args.data_dir, args.dataset_name)
    cache_path = os.path.join(dataset_dir, f'dataset_keywords_{model_tag_from_path(args.model_name)}.pickle')

    if os.path.exists(cache_path):
        dataset = pd.read_pickle(cache_path)
    else:
        dataset = pd.DataFrame(pd.read_pickle(os.path.join(dataset_dir, 'reviews.pickle')))
        encoder = LabelEncoder()
        dataset['user'] = encoder.fit_transform(dataset['user'].tolist()).tolist()
        dataset['item'] = encoder.fit_transform(dataset['item'].tolist()).tolist()

        keywords, keyword_words, text = [], [], []
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        bos_id = tokenizer.bos_token_id

        for row in tqdm(dataset['template'], desc="Processing dataset"):
            kw_tokens = tokenizer(row[0])['input_ids']
            if bos_id is not None and len(kw_tokens) > 0 and kw_tokens[0] == bos_id:
                kw_tokens = kw_tokens[1:]

            txt_tokens = tokenizer(row[2])['input_ids']
            if bos_id is not None and len(txt_tokens) > 0 and txt_tokens[0] == bos_id:
                txt_tokens = txt_tokens[1:]

            keywords.append(kw_tokens)
            keyword_words.append(row[0])
            text.append(txt_tokens + [eos_id])

        dataset['text'] = text
        dataset['keyword'] = keywords
        dataset['keyword_words'] = keyword_words
        dataset = dataset[['user', 'item', 'text', 'keyword', 'keyword_words', 'rating']]
        dataset.to_pickle(cache_path)

    dataset['rating'] = [int(x - 1) for x in dataset['rating'].tolist()]
    return dataset


def load_embedding_object(path):
    if path.endswith(('.pickle', '.pkl')):
        return pd.read_pickle(path)
    return torch.load(path, map_location='cpu')


def build_item_index_mapping(args):
    raw_path = os.path.join(args.data_dir, args.dataset_name, 'reviews.pickle')
    raw_dataset = pd.DataFrame(pd.read_pickle(raw_path))
    encoder = LabelEncoder()
    item_ids = encoder.fit_transform(raw_dataset['item'].tolist())

    raw_to_idx, idx_to_raw = {}, {}
    for raw_item, item_idx in zip(raw_dataset['item'].tolist(), item_ids):
        item_idx = int(item_idx)
        raw_to_idx[raw_item] = item_idx
        raw_to_idx[str(raw_item)] = item_idx
        idx_to_raw[item_idx] = raw_item
    return raw_to_idx, idx_to_raw


def remap_image_embedding_object(image_embeddings, raw_to_idx):
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


def find_item_image(image_dir, raw_item):
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
    if not args.use_multimodal:
        return None

    raw_to_idx, idx_to_raw = build_item_index_mapping(args)
    dataset_dir = os.path.join(args.data_dir, args.dataset_name)

    if args.image_embedding_path:
        image_embeddings = load_embedding_object(args.image_embedding_path)
        image_embeddings = remap_image_embedding_object(image_embeddings, raw_to_idx)
        return MoDLoRA.build_image_embedding_table(image_embeddings, item_num)

    cache_dir = os.path.join(dataset_dir, 'embeddings_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'item_embeddings.pt')

    if os.path.exists(cache_path):
        print(f"Loading cached item image embeddings from {cache_path}")
        image_embeddings = torch.load(cache_path, map_location='cpu')
        image_embeddings = remap_image_embedding_object(image_embeddings, raw_to_idx)
        return MoDLoRA.build_image_embedding_table(image_embeddings, item_num)

    if args.clip_model is None:
        raise ValueError("--clip_model or --image_embedding_path is required when --use_multimodal is enabled.")

    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("PIL and CLIP dependencies are required for --use_multimodal.") from exc

    image_dir = os.path.join(dataset_dir, args.image_dir)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    for param in clip_model.parameters():
        param.requires_grad = False

    item_embeddings = {}
    print(f"Generating item image embeddings with CLIP for {len(idx_to_raw)} items")
    for item_idx, raw_item in tqdm(idx_to_raw.items()):
        image_path = find_item_image(image_dir, raw_item)
        image = Image.open(image_path).convert("RGB") if image_path is not None else Image.new("RGB", (300, 300), (255, 255, 255))
        image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            item_embeddings[item_idx] = clip_model.get_image_features(**image_inputs).cpu()

    torch.save(item_embeddings, cache_path)
    print(f"Item image embeddings saved to {cache_path}")
    del clip_model
    del clip_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return MoDLoRA.build_image_embedding_table(item_embeddings, item_num)


# The following modality-correlation and SVD functions are adapted from PEPLER/analysis.py
# and translated to CIER's batch/model interfaces.
def get_input_embeddings(model):
    if hasattr(model.model, "get_input_embeddings"):
        return model.model.get_input_embeddings()
    return model.model.base_model.get_input_embeddings()


def find_analysis_layer(model):
    for module in model.modules():
        if isinstance(module, MultiModalLoraLayer) and module.x_ui is not None:
            return module
    return None


def cosine_similarity(v1, v2):
    v1 = v1.flatten().to(torch.float32)
    v2 = v2.flatten().to(torch.float32)
    return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()


def pearson_correlation(v1, v2):
    v1_np = v1.flatten().detach().cpu().to(torch.float32).numpy()
    v2_np = v2.flatten().detach().cpu().to(torch.float32).numpy()
    if np.std(v1_np) == 0 or np.std(v2_np) == 0:
        return 0.0
    corr, _ = pearsonr(v1_np, v2_np)
    return corr if not np.isnan(corr) else 0.0


def add_pair_metrics(metrics, name, left, right):
    for b_idx in range(left.size(0)):
        metrics[name]['cos'].append(cosine_similarity(left[b_idx], right[b_idx]))
        metrics[name]['pearson'].append(pearson_correlation(left[b_idx], right[b_idx]))


def analyze_correlation(model, test_dataloader, device, num_batches=10):
    model.eval()
    metrics = {
        'txt_ui': {'cos': [], 'pearson': []},
    }
    if getattr(model, "use_image_lora", False):
        metrics['txt_img'] = {'cos': [], 'pearson': []}
        metrics['ui_img'] = {'cos': [], 'pearson': []}

    print("\n" + "=" * 60)
    print("Start Modality Analysis")

    pad_id = 0
    embeddings_layer = get_input_embeddings(model)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= num_batches:
                break

            input_ids = batch[0].to(device)
            user_id = batch[1].to(device)
            item_id = batch[2].to(device)
            curr_flag = batch[4].to(device)
            rating_inputs = batch[5].to(device)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    model.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id, rating=rating_inputs, curr_flag=curr_flag)
            else:
                model.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id, rating=rating_inputs, curr_flag=curr_flag)

            target_layer = find_analysis_layer(model)
            if target_layer is None:
                continue

            token_embeddings = embeddings_layer(input_ids).to(torch.float32)
            mask = (input_ids != pad_id).unsqueeze(-1).to(torch.float32)
            x_txt = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            batch_size = input_ids.size(0)
            x_ui = target_layer.x_ui[:batch_size].to(torch.float32)
            add_pair_metrics(metrics, 'txt_ui', x_txt, x_ui)

            x_img = getattr(target_layer, "x_img", None)
            if x_img is not None and 'txt_img' in metrics:
                x_img = x_img[:batch_size].to(torch.float32)
                add_pair_metrics(metrics, 'txt_img', x_txt, x_img)
                add_pair_metrics(metrics, 'ui_img', x_ui, x_img)

    print(f"{'Modal Pair':<15} | {'Cosine Sim':<15} | {'Pearson Corr':<15}")
    print("-" * 50)
    for key, val in metrics.items():
        if val['cos']:
            avg_cos = np.mean(val['cos'])
            avg_pearson = np.mean(val['pearson'])
            print(f"{key:<15} | {avg_cos:^15.4f} | {avg_pearson:^15.4f}")
        else:
            print(f"{key:<15} | {'N/A':^15} | {'N/A':^15}")
    print("=" * 60)
    return metrics


def compute_delta_w(lora_A, lora_B, scaling):
    lora_A = lora_A.to(torch.float32)
    lora_B = lora_B.to(torch.float32)
    return (lora_B @ lora_A) * scaling


def analyze_svd_of_lora_weights(model, num_singular_values=100, lora_id=0, output_dir='./analysis_results'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    lora_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, MultiModalLoraLayer)]
    if not lora_layers:
        print("Cannot find MultiModalLoraLayer in model.")
        return {}
    if lora_id >= len(lora_layers):
        raise ValueError(f"lora_id {lora_id} is out of range. Found {len(lora_layers)} LoRA layers.")

    layer_name, lora_layer = lora_layers[lora_id]
    scaling = lora_layer.scaling
    r_val = lora_layer.lora_A_t.shape[0]

    print("\n" + "=" * 80)
    print("Start SVD Analysis for MoDLoRA-based Model")
    print(f"Selected Layer: {layer_name}")
    print(f"Adapter Rank (r): {r_val}")

    delta_w_t = compute_delta_w(lora_layer.lora_A_t.data, lora_layer.lora_B_t.data, scaling)

    ui_scaling = lora_layer.ui_multimodal_scaling.detach()
    delta_w_ui = compute_delta_w(lora_layer.lora_A_ui.data, lora_layer.lora_B_ui.data, scaling * ui_scaling)
    print(f"UI scaling: {ui_scaling.item():.4f}")

    matrices = {
        'Text_$\\Delta$W': delta_w_t,
        'UI_$\\Delta$W': delta_w_ui,
    }
    delta_w_fused = delta_w_t + delta_w_ui

    if getattr(lora_layer, "use_image_lora", False) and lora_layer.lora_A_img is not None:
        image_scaling = lora_layer.image_multimodal_scaling.detach()
        delta_w_img = compute_delta_w(lora_layer.lora_A_img.data, lora_layer.lora_B_img.data, scaling * image_scaling)
        matrices['Image_$\\Delta$W'] = delta_w_img
        delta_w_fused = delta_w_fused + delta_w_img
        print(f"Image scaling: {image_scaling.item():.4f}")
    else:
        print("Image LoRA branch is not enabled; SVD will cover Text and UI only.")

    fused_label = 'Fused_(Text+UI+Image)_$\\Delta$W' if 'Image_$\\Delta$W' in matrices else 'Fused_(Text+UI)_$\\Delta$W'
    fused_matrices = {fused_label: delta_w_fused}
    results = {}

    def plot_matrix_group(matrices_dict, filename, rank_checkpoints, rank_labels):
        plt.figure(figsize=(10, 6))
        for label, delta_w in matrices_dict.items():
            s = torch.linalg.svdvals(delta_w.to(torch.float32)).detach().cpu().numpy()
            num_to_plot = min(len(s), num_singular_values)
            s_plot = s[:num_to_plot]
            indices = np.arange(1, num_to_plot + 1)
            variance_ratio = np.sum(s_plot ** 2) / np.sum(s ** 2) if np.sum(s ** 2) > 0 else 0

            results[label] = {
                'shape': tuple(delta_w.shape),
                f'top_{num_to_plot}_variance_ratio': float(variance_ratio),
            }

            print(f"\n-> SVD Results ({label})")
            print(f"    Shape: {delta_w.shape[0]}x{delta_w.shape[1]}")
            print(f"    First {num_to_plot} singular value ratio: {variance_ratio * 100:.2f}%")
            plt.plot(indices, s_plot, marker='.', linestyle='-', markersize=4, label=label)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Singular Value Index $\\log(k)$', fontsize=10)
        plt.ylabel('Singular Value $\\log(\\sigma_k)$', fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='best')

        ymin, ymax = plt.ylim()
        if ymax <= 0:
            ymax = 1.0

        colors = ['blue', 'red', 'green', 'purple']
        for rc, label, color in zip(rank_checkpoints, rank_labels, colors):
            plt.axvline(x=rc, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
            plt.text(rc * 1.05, ymax * 0.2, label, color=color, fontsize=12, fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Figure saved to: {plot_path}")

    branch_count = len(matrices)
    branch_suffix = 'text_ui_image' if branch_count == 3 else 'text_ui'
    plot_matrix_group(matrices, f'svd_spectrum_{branch_suffix}.png', [r_val], [f'$r={r_val}$'])
    plot_matrix_group(fused_matrices, 'svd_spectrum_fused.png', [r_val * branch_count], [f'${branch_count}r={r_val * branch_count}$'])

    print("=" * 80)
    print(f"SVD figures saved to: {output_dir}")
    return results


def build_model(args, tokenizer, user_num, item_num, image_embeddings, device):
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = f"cuda:{args.devices}"
    model_llm = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model_llm.gradient_checkpointing_enable()

    model = MoDLoRA(
        user_num=user_num,
        item_num=item_num,
        hidden=args.id_hidden,
        llm_hidden=model_llm.config.hidden_size,
        tokenizer=tokenizer,
        model_llm=model_llm,
        r=args.r,
        lora_modules=args.lora_modules,
        image_embeddings=image_embeddings,
        ui_multimodal_scale=args.ui_multimodal_scale,
        image_multimodal_scale=args.image_multimodal_scale,
    ).to(device)
    return model


def load_adapter_state_dict(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(model, args):
    ckpt_path = os.path.join(args.ckpt_dir, args.dataset_name, f'{args.split_index}modlora_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find MoDLoRA checkpoint: {ckpt_path}")
    state = load_adapter_state_dict(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        if not args.use_multimodal and any('img' in key or 'image' in key for key in unexpected):
            print("Warning: checkpoint has image branch weights, but --use_multimodal was not enabled.")


def resolve_cuda_index(devices):
    return 0 if devices < 0 else devices


def parse_args():
    parser = ArgumentParser(description='CIER MoDLoRA multimodal analysis')
    parser.add_argument('--devices', default=-1, type=int, help='Select which GPU to use.')
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--seed', default=5254, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument('--rating_weight', default=0.1, type=float)
    parser.add_argument('--generate_weight', default=1.0, type=float)
    parser.add_argument('--delta', default=0.2, type=float)
    parser.add_argument('--word', default=20, type=int)
    parser.add_argument('--show_train_loss_steps', default=500, type=int)
    parser.add_argument('--id_hidden', default=1024, type=int)
    parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--dataset_name', default='MoviesAndTV', type=str)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--model_name', default='../autodl-fs/Qwen2.5-7B/', type=str)
    parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    parser.add_argument('--log_dir', default='./log/', type=str)
    parser.add_argument('--log_name', default='llama.log', type=str)
    parser.add_argument('--split_index', default='1', type=str)
    parser.add_argument('--lora_modules', type=int, default=2)
    parser.add_argument('--r', type=int, default=4)
    parser.add_argument('--use_multimodal', action='store_true', help='Enable item-image LoRA branch for analysis.')
    parser.add_argument('--clip_model', default=None, type=str)
    parser.add_argument('--image_dir', default='images', type=str)
    parser.add_argument('--image_embedding_path', default=None, type=str)
    parser.add_argument('--ui_multimodal_scale', default=2.0, type=float)
    parser.add_argument('--image_multimodal_scale', default=2.0, type=float)
    parser.add_argument('--num_batches', default=10, type=int)
    parser.add_argument('--num_singular_values', default=100, type=int)
    parser.add_argument('--lora_id', default=0, type=int)
    parser.add_argument('--output_dir', default='./analysis_results', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.devices = resolve_cuda_index(args.devices)
    device = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")

    user_num, item_num = infer_user_item_num(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_or_build_dataset(args, tokenizer)

    _, _, test_dataset = dataset_split(dataset, args.split_index, args)
    test_set = MyDataset(test_dataset)
    collate_test = MyCollater(1, args.word)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_test, shuffle=False)

    image_embeddings = build_multimodal_image_embeddings(args, item_num, device)
    model = build_model(args, tokenizer, user_num, item_num, image_embeddings, device)
    load_checkpoint(model, args)

    print("\n=== [Start Analysis of Subspace Decoupling and Spectral Extension for CIER] ===")
    analyze_correlation(model, test_dataloader, device, num_batches=args.num_batches)
    analyze_svd_of_lora_weights(
        model,
        num_singular_values=args.num_singular_values,
        lora_id=args.lora_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
