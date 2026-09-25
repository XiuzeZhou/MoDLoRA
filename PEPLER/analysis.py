"""Modality correlation and full-spectrum three-branch SVD for PEPLER MoDLoRA."""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from module import MoDLoRA, MultiModalLoraLayer
from utils import DataLoader, Batchify, now_time


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


def pairwise_correlations(first, second):
    """Per-sample correlations; zero/constant vectors receive a score of zero."""
    first, second = first.float(), second.float()
    cosine = F.cosine_similarity(first, second, dim=-1)
    centered_first = first - first.mean(dim=-1, keepdim=True)
    centered_second = second - second.mean(dim=-1, keepdim=True)
    pearson = F.cosine_similarity(centered_first, centered_second, dim=-1)
    return cosine.cpu().tolist(), pearson.cpu().tolist()


@torch.no_grad()
def analyze_correlation(model, test_data, device, num_batches=16, lora_id=0,
                        output_dir=None, layer_id=None, module_name='q_proj'):
    """Compare inputs to the selected LoRA layer, not its projected outputs.

    Text is the attention-masked mean of the actual layer input (prompt + review).
    UI/image inputs are the normalized contexts installed by model.set_Q().
    These descriptive input correlations do not by themselves prove that the
    learned LoRA output subspaces are orthogonal.
    """
    if num_batches <= 0:
        raise ValueError('num_batches must be positive.')
    layer_name, layer = select_lora_layer(model, lora_id, layer_id, module_name)
    if not context_branches_active(layer):
        raise ValueError(f'{layer_name} skips UI/image branches because in_features != hidden_size. '
                         'Choose an active layer for correlation, or use --analysis svd.')
    pairs = [('txt_ui', 'txt', 'ui'), ('txt_img', 'txt', 'img'), ('ui_img', 'ui', 'img')]
    metrics = {pair: {'cos': [], 'pearson': []} for pair, _, _ in pairs}
    original_modes = [(module, module.training) for module in model.modules()]
    original_step = test_data.step
    original_contexts = [(m, m.x_ui, m.x_img) for m in model.modules()
                         if isinstance(m, MultiModalLoraLayer)]
    captured = {}
    attention_mask = None

    def capture_inputs(_module, inputs):
        hidden = inputs[0].float()
        if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
            raise ValueError('Expected layer inputs shaped (batch, sequence, hidden).')
        # Use the supplied mask: a genuine EOS token can share the padding ID.
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        captured['txt'] = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()
        for key, context in [('ui', layer.x_ui), ('img', layer.x_img)]:
            if context is None or context.shape != captured['txt'].shape:
                raise ValueError(f'{layer_name}: missing or incorrectly shaped {key} context.')
            captured[key] = context.float().cpu()

    hook = layer.register_forward_pre_hook(capture_inputs)
    model.eval()
    test_data.step = 0
    print(f'\nModality input correlation at {layer_name} (prompt + review)')
    try:
        for _ in range(min(num_batches, test_data.total_step)):
            user, item, _, input_ids, mask, *_ = test_data.next_batch()
            user, item = user.to(device), item.to(device)
            attention_mask = mask.to(device)
            model.set_Q(*model.build_lora_context(user, item))
            captured.clear()
            # The backbone executes the selected layer without allocating LM logits.
            model.model.base_model(
                input_ids=input_ids.to(device), attention_mask=attention_mask,
                use_cache=False, output_hidden_states=False, output_attentions=False,
            )
            if not captured:
                raise RuntimeError(f'The forward pass did not execute {layer_name}.')
            for pair, first, second in pairs:
                cos, pearson = pairwise_correlations(captured[first], captured[second])
                metrics[pair]['cos'].extend(cos)
                metrics[pair]['pearson'].extend(pearson)
    finally:
        hook.remove()
        test_data.step = original_step
        for module, x_ui, x_img in original_contexts:
            module.x_ui, module.x_img = x_ui, x_img
        for module, training in original_modes:
            module.training = training

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
                       'representation': 'layer inputs; text = masked mean over prompt + review',
                       'metrics': metrics, 'summary': summary}, file, indent=2)
        print(f'Correlation results saved to: {output_path}')
    return metrics


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
        branches.update({
            'ui': (layer.lora_A_ui, layer.lora_B_ui,
                   layer.scaling * layer.ui_multimodal_scaling.item()),
            'image': (layer.lora_A_img, layer.lora_B_img,
                      layer.scaling * layer.image_multimodal_scaling.item()),
        })
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


def resolve_checkpoint(checkpoint, dataset_name):
    """Accept a file, main.py's checkpoint directory, or the legacy root directory."""
    if os.path.isfile(checkpoint):
        return checkpoint
    candidates = [os.path.join(checkpoint, 'model.pt'),
                  os.path.join(checkpoint, dataset_name, 'model.pt')]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError('Checkpoint not found. Pass --checkpoint with the trained .pt file '
                            f'or its directory. Tried: {candidates}')


def load_analysis_checkpoint(model, model_path):
    """Frozen backbone weights/buffers may be absent from main.py's saved state."""
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = sorted(trainable_names - state.keys())
    unexpected = sorted(state.keys() - model.state_dict().keys())
    if missing or unexpected:
        raise ValueError('Checkpoint does not match the current MoDLoRA architecture. '
                         f'Missing trainable parameters: {missing[:10]}; '
                         f'unexpected parameters: {unexpected[:10]}. '
                         'Check the backbone, k, r, mlp_size and lora_modules; old MoE '
                         'checkpoints cannot be loaded as three-branch MoDLoRA.')
    model.load_state_dict(state, strict=False)
    print(now_time() + f'Loaded all trainable parameters from {model_path}')


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze text/UI/image LoRA branches in PEPLER MoDLoRA')
    parser.add_argument('-dataset_name', '--dataset_name', default='ClothingShoesAndJewelry')
    parser.add_argument('-data_path', '--data_path', default=None)
    parser.add_argument('-index_dir', '--index_dir', default=None)
    parser.add_argument('-llm_model', '--llm_model', default='/root/autodl-fs/Qwen2.5-7B/')
    parser.add_argument('-clip_model', '--clip_model', default='../llms/clip-vit-base-patch32/')
    parser.add_argument('-checkpoint', '--checkpoint', default='./checkpoints/',
                        help='trained .pt file, checkpoint directory, or dataset checkpoint root')
    parser.add_argument('-batch_size', '--batch_size', type=positive_int, default=16)
    parser.add_argument('-words', '--words', type=positive_int, default=20)
    parser.add_argument('-mlp_size', '--mlp_size', type=positive_int, default=400)
    parser.add_argument('-k', '--k', type=positive_int, default=768)
    parser.add_argument('-r', '--r', type=positive_int, default=8)
    parser.add_argument('-lora_modules', '--lora_modules', type=positive_int, default=7)
    parser.add_argument('-image_dir', '--image_dir', default='images')
    parser.add_argument('-ui_multimodal_scale', '--ui_multimodal_scale', type=float, default=2.0)
    parser.add_argument('-image_multimodal_scale', '--image_multimodal_scale', type=float, default=2.0)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('-cuda', '--cuda', dest='cuda', action='store_true')
    device_group.add_argument('-cpu', '--cpu', dest='cuda', action='store_false')
    parser.set_defaults(cuda=torch.cuda.is_available())
    parser.add_argument('-num_batches', '--num_batches', type=positive_int, default=16,
                        help='maximum number of test batches used for correlation')
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument('-lora_id', '--lora_id', default=None, type=int,
                             help='legacy flattened LoRA index in named_modules(), not the Transformer layer number')
    layer_group.add_argument('-layer_id', '--layer_id', default=None, type=int,
                             help='zero-based Transformer layer number, e.g. 12 selects layers.12')
    parser.add_argument('-module_name', '--module_name', default='q_proj',
                        choices=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                        help='projection to analyze with --layer_id (default: q_proj); must contain trained LoRA weights')
    parser.add_argument('-num_singular_values', '--num_singular_values', type=positive_int, default=10,
                        help='number of full-spectrum points to plot, including the tail beyond the rank')
    parser.add_argument('-analysis', '--analysis', choices=['both', 'correlation', 'svd'], default='both',
                        help='both (default): similarity + SVD; correlation: similarity only; svd: weights only')
    parser.add_argument('-output_dir', '--output_dir', default='./analysis_results')
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        parser.error('CUDA was requested but is not available.')
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
    device = torch.device('cuda' if args.cuda else 'cpu')
    model_path = resolve_checkpoint(args.checkpoint, args.dataset_name)
    data_path = args.data_path or os.path.join('../data', args.dataset_name, 'reviews.pickle')
    index_dir = args.index_dir or os.path.join(os.path.dirname(data_path), '1')

    # Keep special-token setup identical to main.py, including non-Qwen backbones.
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side='left',
                                              spaces_between_special_tokens=False)
    bos, eos, pad = '<bos>', '<eos>', '<pad>'
    if any(name in args.llm_model.lower() for name in ('llama', 'qwen', 'mistral', 'gemma')):
        bos = tokenizer.bos_token or '<s>'
        eos = tokenizer.eos_token or '</s>'
        pad = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.pad_token = pad
    tokenizer.add_special_tokens({'bos_token': bos, 'eos_token': eos, 'pad_token': pad})
    corpus = DataLoader(data_path, index_dir, tokenizer, args.words, clip_path=args.clip_model,
                        image_dir=args.image_dir, device=device)
    model = MoDLoRA.from_pretrained(
        args.llm_model, nuser=len(corpus.user_dict), nitem=len(corpus.item_dict),
        k=args.k, r=args.r, mlp_size=args.mlp_size, lora_modules=args.lora_modules,
        image_embeddings=corpus.image_embeddings,
        ui_multimodal_scale=args.ui_multimodal_scale,
        image_multimodal_scale=args.image_multimodal_scale,
        device_map={'': str(device)},
    )
    model.resize_token_embeddings(len(tokenizer))
    load_analysis_checkpoint(model, model_path)
    model.to(device)
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.analysis in ('both', 'correlation'):
        if not corpus.test:
            raise ValueError('The test split is empty; correlation requires test samples.')
        test_data = Batchify(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer,
                            bos, eos, args.words, args.batch_size, corpus.max_rating, corpus.min_rating)
        analyze_correlation(model, test_data, device, num_batches=args.num_batches,
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


if __name__ == '__main__':
    main()
