"""Modality correlation and full-spectrum three-branch SVD for CIER MoDLoRA."""

import json
import os
import random
from argparse import ArgumentParser, ArgumentTypeError
from contextlib import nullcontext
from itertools import islice

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
class AnalysisCollater(MyCollater):
    """Keep CIER's batch fields and append a mask based on true sequence lengths."""

    def __call__(self, data):
        batch = super().__call__(data)
        input_ids, _, _, _, flags, _ = batch
        lengths = torch.tensor([
            min(len(row['text' if flag else 'keyword']), input_ids.size(1))
            for row, flag in zip(data, flags.tolist())
        ])
        # Token ID 0 can be a real vocabulary token as well as CIER's padding ID.
        mask = torch.arange(input_ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        return (*batch, mask)


def pairwise_correlations(first, second):
    """Per-sample cosine/Pearson; undefined zero-norm cases receive zero."""
    first, second = first.float(), second.float()
    cosine = F.cosine_similarity(first, second, dim=-1)
    centered_first = first - first.mean(dim=-1, keepdim=True)
    centered_second = second - second.mean(dim=-1, keepdim=True)
    pearson = F.cosine_similarity(centered_first, centered_second, dim=-1)
    return cosine.cpu().tolist(), pearson.cpu().tolist()


@torch.no_grad()
def analyze_correlation(model, test_dataloader, device, num_batches=10, lora_id=0,
                        output_dir=None, layer_id=None, module_name='q_proj'):
    """Compare actual inputs to the selected LoRA layer, as in PEPLER.

    Text is the masked mean of the layer input over the CIER prompt and review.
    UI/image are the contexts installed by CIER's get_embedding(), without
    adding normalization that is absent from the trained model. These input
    correlations do not measure orthogonality of learned output subspaces.
    Use AnalysisCollater to distinguish real token ID 0 from padding; legacy
    six-field batches fall back to CIER's token-ID-0 padding convention.
    """
    if num_batches <= 0:
        raise ValueError('num_batches must be positive.')
    device = torch.device(device)
    layer_name, layer = select_lora_layer(model, lora_id, layer_id, module_name)
    if not context_branches_active(layer):
        raise ValueError(f'{layer_name} skips UI/image branches because in_features != hidden_size. '
                         'Choose an active layer for correlation, or use --analysis svd.')
    pairs = [('txt_ui', 'txt', 'ui')]
    if layer.use_image_lora:
        pairs.extend([('txt_img', 'txt', 'img'), ('ui_img', 'ui', 'img')])
    metrics = {pair: {'cos': [], 'pearson': []} for pair, _, _ in pairs}
    original_modes = [(module, module.training) for module in model.modules()]
    original_contexts = [(module, module.x_ui, module.x_img) for module in model.modules()
                         if isinstance(module, MultiModalLoraLayer)]
    collater = getattr(test_dataloader, 'collate_fn', None)
    original_step = getattr(collater, 'cur_step', None)
    captured = {}
    attention_mask = None

    def capture_inputs(_module, inputs):
        hidden = inputs[0].float()
        if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
            raise ValueError('Expected layer inputs shaped (batch, sequence, hidden).')
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        captured['txt'] = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()
        for key, context in [('ui', layer.x_ui)] + ([('img', layer.x_img)] if layer.use_image_lora else []):
            if context is None or context.shape != captured['txt'].shape:
                raise ValueError(f'{layer_name}: missing or incorrectly shaped {key} context.')
            captured[key] = context.float().cpu()

    hook = layer.register_forward_pre_hook(capture_inputs)
    model.eval()
    print(f'\nModality input correlation at {layer_name} (CIER prompt + review)')
    try:
        for batch in islice(test_dataloader, num_batches):
            input_ids, user_id, item_id, _, curr_flag, rating_inputs = batch[:6]
            review_mask = batch[6] if len(batch) > 6 else input_ids.ne(0)
            if review_mask.shape != input_ids.shape:
                raise ValueError('Review attention mask must have the same shape as input_ids.')
            input_ids = input_ids.to(device)
            amp = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with amp:
                inputs_embeds = model.get_embedding(
                    input_ids=input_ids, user_id=user_id.to(device), item_id=item_id.to(device),
                    rating=rating_inputs.to(device), curr_flag=curr_flag.to(device),
                )
                prompt_length = inputs_embeds.size(1) - input_ids.size(1)
                if prompt_length < 0:
                    raise ValueError('CIER inputs_embeds must contain the prompt and review.')
                prompt_mask = torch.ones((input_ids.size(0), prompt_length), device=device, dtype=torch.bool)
                attention_mask = torch.cat([prompt_mask, review_mask.to(device=device, dtype=torch.bool)], dim=1)
                captured.clear()
                # Execute the selected layer without allocating vocabulary-sized LM logits.
                model.model.base_model(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    use_cache=False, output_hidden_states=False, output_attentions=False,
                )
            if not captured:
                raise RuntimeError(f'The forward pass did not execute {layer_name}.')
            for pair, first, second in pairs:
                cosine, pearson = pairwise_correlations(captured[first], captured[second])
                metrics[pair]['cos'].extend(cosine)
                metrics[pair]['pearson'].extend(pearson)
    finally:
        hook.remove()
        for module, x_ui, x_img in original_contexts:
            module.x_ui, module.x_img = x_ui, x_img
        for module, training in original_modes:
            module.training = training
        if original_step is not None:
            collater.cur_step = original_step

    if not metrics['txt_ui']['cos']:
        raise ValueError('No samples were available for modality correlation.')
    summary = {
        pair: {'samples': len(values['cos']), 'mean_cosine': float(np.mean(values['cos'])),
               'mean_pearson': float(np.mean(values['pearson']))}
        for pair, values in metrics.items()
    }
    print(f"{'Modal pair':<15} | {'Cosine':>12} | {'Pearson':>12} | {'Samples':>8}")
    for pair, values in summary.items():
        print(f"{pair:<15} | {values['mean_cosine']:12.4f} | "
              f"{values['mean_pearson']:12.4f} | {values['samples']:8d}")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'correlation_results.json')
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump({'layer': layer_name,
                       'representation': 'layer inputs; text = masked mean over CIER prompt + review',
                       'metrics': metrics, 'summary': summary}, file, indent=2)
        print(f'Correlation results saved to: {output_path}')
    return metrics


def select_lora_layer(model, lora_id=0, layer_id=None, module_name='q_proj'):
    """Select by Transformer block/projection, or the legacy flattened LoRA index."""
    layers = [(name, layer) for name, layer in model.named_modules()
              if isinstance(layer, MultiModalLoraLayer)]
    if not layers:
        raise ValueError('No MultiModalLoraLayer found; use a MoDLoRA checkpoint.')
    if layer_id is not None:
        if layer_id < 0:
            raise ValueError('layer_id must be nonnegative.')
        block_path = f'.layers.{layer_id}.'
        matches = [(name, layer) for name, layer in layers
                   if block_path in f'.{name}' and name.rsplit('.', 1)[-1] == module_name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f'Multiple LoRA modules match layer_id={layer_id}, module_name={module_name}: '
                             f'{[name for name, _ in matches]}. Use --lora_id to select one explicitly.')
        available = [name for name, _ in layers]
        raise ValueError(f'No LoRA module matches layers.{layer_id}.*.{module_name}. '
                         'Check the Transformer layer number and the trained lora_modules setting. '
                         f'Available LoRA modules (first 12 of {len(available)}): {available[:12]}')
    if not 0 <= lora_id < len(layers):
        raise ValueError(f'lora_id must be between 0 and {len(layers) - 1}, got {lora_id}.')
    return layers[lora_id]


def context_branches_active(layer):
    # This is the dimension check used by MultiModalLoraLayer._context_lora.
    return layer.base_layer.in_features == layer.hidden_size


def compute_delta_w(lora_A, lora_B, scaling):
    """Materialize scaling * B @ A in CPU FP32 for full-spectrum analysis."""
    a = lora_A.detach().to(device='cpu', dtype=torch.float32)
    b = lora_B.detach().to(device='cpu', dtype=torch.float32)
    return (b @ a) * float(scaling)


def plot_spectra(results, keys, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    positive_values = [s for key in keys for s in results[key]['plotted_singular_values'] if s > 0]
    has_positive_values = bool(positive_values)
    zero_floor = min(positive_values) * 0.1 if has_positive_values else 0.0
    zeros_clipped = False
    marked_ranks = set()
    for key in keys:
        result = results[key]
        values = np.asarray(result['plotted_singular_values'])
        # Keep exact zeros visible on log axes; the JSON retains the raw values.
        plotted = np.maximum(values, zero_floor) if has_positive_values else values
        zeros_clipped |= has_positive_values and bool(np.any(values == 0))
        label = rf"{key.capitalize()}_$\Delta$W"
        if not np.any(values > 0):
            label += ' [zero matrix]'
        line, = ax.plot(np.arange(1, len(values) + 1), plotted, '.-', markersize=3, label=label)
        rank = result['rank_upper_bound']
        # Modalities with the same rank share one marker and one label.
        if rank not in marked_ranks:
            marked_ranks.add(rank)
            rank_label = f'$r = {rank}$'
            if key == 'fused':
                branch_ranks = [value['rank_upper_bound'] for name, value in results.items()
                                if name != 'fused']
                if len(branch_ranks) > 1 and len(set(branch_ranks)) == 1 and sum(branch_ranks) == rank:
                    rank_label = rf'$\sum r_m = {rank}$'
                else:
                    rank_label = rf'$r_{{\mathrm{{fused}}}} = {rank}$'
            ax.axvline(rank, color=line.get_color(), linestyle='--', alpha=0.6)
            ax.annotate(
                rank_label, xy=(rank, 0.95), xycoords=ax.get_xaxis_transform(),
                xytext=(6, 0), textcoords='offset points',
                color=line.get_color(), fontsize=12, ha='left', va='top',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 2},
            )
    ax.set_xscale('log')
    if has_positive_values:
        ax.set_yscale('log')
    if zeros_clipped:
        ax.text(0.02, 0.02, f'Exact zeros displayed at {zero_floor:.1e}',
                transform=ax.transAxes, fontsize=8)
    ax.set_xlabel('Singular value index k')
    ax.set_ylabel('Singular value')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Figure saved to: {output_path}')


@torch.no_grad()
def analyze_svd_of_lora_weights(model, num_singular_values=128, lora_id=0,
                               output_dir='./analysis_results', layer_id=None, module_name='q_proj'):
    """Analyze active branch maps and their fused map on concatenated inputs.

    The adapter adds Dt @ x_t + Dui @ x_ui + Dimg @ x_img (column notation).
    Its fused operator is [Dt Dui Dimg], NOT Dt + Dui + Dimg, since the inputs
    differ. It is not a single weight update that can be merged into W0.
    Its rank is at most min(output_dim, total_input_dim, sum(branch_ranks)).

    Full CPU FP32 SVD retains the numerical tail beyond the theoretical rank,
    so the plot shows the drop and its tail up to num_singular_values. Unlike
    reduced QR, this requires materializing the dense matrices and costs more
    CPU time/memory. Tiny tail values are roundoff, not additional effective rank.
    """
    if num_singular_values <= 0:
        raise ValueError('num_singular_values must be positive.')
    layer_name, layer = select_lora_layer(model, lora_id, layer_id, module_name)
    os.makedirs(output_dir, exist_ok=True)
    branches = {'text': (layer.lora_A_t, layer.lora_B_t, layer.scaling)}
    if context_branches_active(layer):
        branches['ui'] = (layer.lora_A_ui, layer.lora_B_ui,
                          layer.scaling * layer.ui_multimodal_scaling.item())
        if layer.use_image_lora:
            branches['image'] = (layer.lora_A_img, layer.lora_B_img,
                                 layer.scaling * layer.image_multimodal_scaling.item())
        else:
            print(f'{layer_name}: image LoRA is disabled; analyzing text/UI only.')
    else:
        print(f'{layer_name}: UI/image branches are inactive; analyzing text only.')

    print(f'\nFull FP32 LoRA weight SVD at {layer_name}')
    print('Values beyond the theoretical rank show the floating-point numerical tail.')
    results = {}

    def record(key, delta_w, rank_upper_bound, scale):
        # Do not reduce to an r x r matrix: doing so discards the tail we plot.
        values = torch.linalg.svdvals(delta_w).double().numpy()
        shape = tuple(delta_w.shape)
        count = min(num_singular_values, len(values))
        energy = float(np.sum(values ** 2))
        ratio = float(np.sum(values[:count] ** 2) / energy) if energy else 0.0
        tolerance = max(shape) * np.finfo(np.float32).eps * values[0]
        results[key] = {
            'shape': list(shape),
            'scaling': float(scale),
            'rank_upper_bound': min(*shape, rank_upper_bound),
            'numerical_rank': int(np.count_nonzero(values > tolerance)),
            'rank_tolerance': float(tolerance),
            'singular_values': values.tolist(),
            'implicit_zero_count': 0,
            'plotted_singular_values': values[:count].tolist(),
            f'top_{count}_energy_ratio': ratio,
        }
        print(f"{key:<8} shape={shape}, scale={float(scale):.4f}, "
              f"rank={results[key]['numerical_rank']} <= {results[key]['rank_upper_bound']}, "
              f'top-{count} energy={ratio:.2%}')

    # Preallocate the fused map and fill it one branch at a time to avoid holding
    # all three dense branch matrices plus a second concatenated copy in memory.
    output_dim = layer.lora_B_t.shape[0]
    input_dim = sum(a.shape[1] for a, _, _ in branches.values())
    fused = torch.empty((output_dim, input_dim), dtype=torch.float32, device='cpu')
    offset = 0
    for key, (a, b, scale) in branches.items():
        delta_w = compute_delta_w(a, b, scale)
        record(key, delta_w, a.shape[0], scale)
        fused[:, offset:offset + a.shape[1]] = delta_w
        offset += a.shape[1]
        del delta_w
    record('fused', fused, sum(a.shape[0] for a, _, _ in branches.values()), 1.0)
    del fused
    plot_spectra(results, list(branches), os.path.join(output_dir, 'svd_spectrum_branches.png'))
    plot_spectra(results, ['fused'], os.path.join(output_dir, 'svd_spectrum_fused.png'))
    with open(os.path.join(output_dir, 'svd_results.json'), 'w', encoding='utf-8') as file:
        json.dump({'layer': layer_name,
                   'operator': '[' + ', '.join(f'Delta_W_{key}' for key in branches) + ']',
                   'svd_method': 'dense_float32',
                   'tail_note': 'Values beyond the theoretical rank are floating-point roundoff.',
                   'results': results}, file, indent=2)
    return results


def build_model(args, tokenizer, user_num, item_num, image_embeddings, device):
    model_kwargs = {"torch_dtype": torch.bfloat16 if device.type == 'cuda' else torch.float32}
    if device.type == 'cuda':
        model_kwargs["device_map"] = str(device)
    model_llm = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

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
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        checkpoint_dir = os.path.join(args.ckpt_dir, args.dataset_name)
        candidates = [os.path.join(checkpoint_dir, f'{args.split_index}{suffix}_model.pth')
                      for suffix in ('modlora', 'uiadapter')]
        ckpt_path = next((path for path in candidates if os.path.isfile(path)), candidates[0])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find MoDLoRA checkpoint: {ckpt_path}. "
                                "Pass --checkpoint for an explicit .pth file. "
                                "Three-branch analysis requires training with --use_modlora --use_multimodal.")
    state = load_adapter_state_dict(ckpt_path, map_location="cpu")
    # main.py saves trainable parameters only. Missing frozen backbone weights
    # and image buffers are expected; missing learned branch weights are not.
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = sorted(trainable_names - state.keys())
    unexpected = sorted(state.keys() - model.state_dict().keys())
    if missing or unexpected:
        raise ValueError('Checkpoint does not match the current CIER MoDLoRA architecture. '
                         f'Missing trainable parameters: {missing[:10]}; '
                         f'unexpected parameters: {unexpected[:10]}. '
                         'Check the backbone, r, id_hidden, lora_modules and image-branch setting. '
                         'For a legacy text/UI checkpoint use --no_multimodal; '
                         'three-branch SVD requires trained image LoRA weights.')
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")


def resolve_cuda_index(devices):
    return 0 if devices < 0 else devices


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise ArgumentTypeError('must be a positive integer')
    return value


def parse_args():
    parser = ArgumentParser(description='Analyze text/UI/image LoRA branches in CIER MoDLoRA')
    parser.add_argument('--devices', default=-1, type=int, help='Select which GPU to use.')
    parser.add_argument('--batch_size', default=40, type=positive_int)
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
    parser.add_argument('--dataset_name', default='ClothingShoesAndJewelry', type=str)
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--model_name', default='/root/autodl-fs/Qwen2.5-7B/', type=str)
    parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    parser.add_argument('--checkpoint', default=None,
                        help='explicit trained MoDLoRA .pth file; overrides ckpt_dir and split_index')
    parser.add_argument('--log_dir', default='./log/', type=str)
    parser.add_argument('--log_name', default='llama.log', type=str)
    parser.add_argument('--split_index', default='1', type=str)
    parser.add_argument('--lora_modules', type=int, choices=range(1, 8), default=2)
    parser.add_argument('--r', type=positive_int, default=4)
    modality_group = parser.add_mutually_exclusive_group()
    modality_group.add_argument('--use_multimodal', dest='use_multimodal', action='store_true',
                                help='analyze text/UI/image branches (default)')
    modality_group.add_argument('--no_multimodal', dest='use_multimodal', action='store_false',
                                help='analyze a legacy text/UI checkpoint without image LoRA')
    parser.set_defaults(use_multimodal=True)
    parser.add_argument('--clip_model', default=None, type=str)
    parser.add_argument('--image_dir', default='images', type=str)
    parser.add_argument('--image_embedding_path', default=None, type=str)
    parser.add_argument('--ui_multimodal_scale', default=2.0, type=float)
    parser.add_argument('--image_multimodal_scale', default=2.0, type=float)
    parser.add_argument('--num_batches', default=16, type=positive_int,
                        help='maximum test batches used for modality cosine/Pearson correlations')
    parser.add_argument('--num_singular_values', default=4, type=positive_int,
                        help='number of full-spectrum points to plot, including the numerical tail')
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument('--lora_id', default=None, type=int,
                             help='legacy flattened LoRA index in named_modules(), not the Transformer layer number')
    layer_group.add_argument('--layer_id', default=None, type=int,
                             help='zero-based Transformer layer number, e.g. 12 selects layers.12')
    parser.add_argument('--module_name', default='q_proj',
                        choices=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                        help='projection to analyze with --layer_id (default: q_proj); must contain trained LoRA weights')
    parser.add_argument('--analysis', choices=['both', 'correlation', 'svd'], default='both',
                        help='both (default): similarity + SVD; correlation: similarity only; svd: weights only')
    parser.add_argument('--output_dir', default='./analysis_results', type=str)
    args = parser.parse_args()
    if args.lora_id is None:
        args.lora_id = 0
    if args.lora_id < 0:
        parser.error('lora_id must be nonnegative.')
    if args.layer_id is not None and args.layer_id < 0:
        parser.error('layer_id must be nonnegative.')
    if args.layer_id is None and args.module_name != 'q_proj':
        parser.error('--module_name requires --layer_id.')
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.devices = resolve_cuda_index(args.devices)
    device = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")

    user_num, item_num = infer_user_item_num(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    image_embeddings = build_multimodal_image_embeddings(args, item_num, device)
    model = build_model(args, tokenizer, user_num, item_num, image_embeddings, device)
    load_checkpoint(model, args)
    model.eval()

    print("\n=== [CIER MoDLoRA Modality and Weight-Spectrum Analysis] ===")
    if args.analysis in ('both', 'correlation'):
        dataset = load_or_build_dataset(args, tokenizer)
        _, _, test_dataset = dataset_split(dataset, args.split_index, args)
        test_set = MyDataset(test_dataset)
        if not len(test_set):
            raise ValueError('The test split is empty; correlation requires test samples.')
        collate_test = AnalysisCollater(1, args.word)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_test, shuffle=False)
        analyze_correlation(model, test_dataloader, device, num_batches=args.num_batches,
                            lora_id=args.lora_id, output_dir=args.output_dir,
                            layer_id=args.layer_id, module_name=args.module_name)
    if args.analysis in ('both', 'svd'):
        analyze_svd_of_lora_weights(
            model,
            num_singular_values=args.num_singular_values,
            lora_id=args.lora_id,
            output_dir=args.output_dir,
            layer_id=args.layer_id,
            module_name=args.module_name,
        )


if __name__ == "__main__":
    main()
