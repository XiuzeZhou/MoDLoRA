from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import math

class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):  # (batch_size, emsize)
        rating = torch.sum(user * item, 1)  # (batch_size,)
        return rating


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class NeuMF_Predictor(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=0, dropout=0.3):
        super(NeuMF_Predictor, self).__init__()
        self.first_layer = nn.Linear(emsize * 2, hidden_size, dtype=torch.float32)
        self.last_layer = nn.Linear(hidden_size, 1, dtype=torch.float32)
        
        layer = nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        self.layers = _get_clones(layer, num_layers)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.mf_weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.mlp_weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.first_layer.weight)
        if self.first_layer.bias is not None:
            self.first_layer.bias.data.zero_()
            
        nn.init.normal_(self.last_layer.weight, mean=0.0, std=0.01)
        if self.last_layer.bias is not None:
            self.last_layer.bias.data.zero_()
            
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, p_u, q_i):  
        p_u = p_u.to(torch.float32)
        q_i = q_i.to(torch.float32)
        
        # 1. MF Branch
        mf_output = torch.sum(p_u * q_i, dim=1)
        
        # 2. MLP Branch
        ui_cat = torch.cat([p_u, q_i], dim=1)
        ui_cat = self.dropout(ui_cat)
        hidden = self.sigmoid(self.first_layer(ui_cat))  
        hidden = self.dropout(hidden)
        
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  
            hidden = self.dropout(hidden)
            
        mlp_output = torch.squeeze(self.last_layer(hidden))  
        
        rating = self.mf_weight * mf_output + self.mlp_weight * mlp_output
        
        return self.sigmoid(rating)


class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=0, is_sigmoid=False, dtype=torch.bfloat16):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize, hidden_size, dtype=dtype)
        self.last_layer = nn.Linear(hidden_size, 1, dtype=dtype)
        layer = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.is_sigmoid = is_sigmoid

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.first_layer.weight, mode='fan_in', nonlinearity='relu')
        if self.first_layer.bias is not None:
            self.first_layer.bias.data.zero_()
            
        nn.init.normal_(self.last_layer.weight, mean=0.0, std=0.001)
        if self.last_layer.bias is not None:
            self.last_layer.bias.data.zero_()
            
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='sigmoid')
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, x):  
        hidden = self.relu(self.first_layer(x))  
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  
        rating = torch.squeeze(self.last_layer(hidden))  
        if self.is_sigmoid:
            rating = self.sigmoid(rating)
        return rating


class LoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_layer = base_layer

        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.0)
        
        # text LoRA (A_t, B_t)
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A_t = nn.Parameter(torch.randn(r, in_features, dtype=dtype))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout)
    
    def forward(self, x):
        # y = x_t * W0 + x_t * B_t * A_t
        lora_t = self.lora_dropout(x) @ self.lora_A_t.transpose(0, 1) @ self.lora_B_t.transpose(0, 1) * self.scaling
        result = self.base_layer(x) + lora_t
        return result
    

class AdaLoraLayer(LoraLayer):
    """Adaptive-rank SVD adapter for the concatenated multimodal baseline."""

    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__(base_layer, hidden_size=hidden_size, dtype=dtype, **kwargs)
        nn.init.normal_(self.lora_A_t, std=0.02)
        nn.init.normal_(self.lora_B_t, std=0.02)
        self.lora_E_t = nn.Parameter(torch.zeros(self.lora_A_t.size(0), 1, dtype=torch.float32))

    def forward(self, x):
        update = (
            self.lora_dropout(x)
            @ (self.lora_A_t.to(x.dtype) * self.lora_E_t.to(x.dtype)).transpose(0, 1)
            @ self.lora_B_t.transpose(0, 1).to(x.dtype)
            * self.scaling
        )
        return self.base_layer(x) + update


class MultiModalLoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_layer = base_layer

        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.0)
        ui_multimodal_scale = kwargs.pop("ui_multimodal_scale", 1.0)
        image_multimodal_scale = kwargs.pop("image_multimodal_scale", 1.0)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A_t = nn.Parameter(torch.randn(r, in_features, dtype=dtype))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))

        self.lora_A_ui = nn.Parameter(torch.randn(r, in_features, dtype=dtype))
        self.lora_B_ui = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))

        self.lora_A_img = nn.Parameter(torch.randn(r, in_features, dtype=dtype))
        self.lora_B_img = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))

        self.scaling = lora_alpha / r
        self.ui_multimodal_scaling = nn.Parameter(torch.tensor(float(ui_multimodal_scale), dtype=torch.float32))
        self.image_multimodal_scaling = nn.Parameter(torch.tensor(float(image_multimodal_scale), dtype=torch.float32))
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.x_ui = None
        self.x_img = None

    def _align_context(self, context, batch_size):
        if context is None:
            return None
        if context.size(0) == batch_size:
            return context
        if batch_size % context.size(0) == 0:
            repeat = batch_size // context.size(0)
            return context.repeat_interleave(repeat, dim=0)
        context = context[:batch_size]
        return context if context.size(0) == batch_size else None

    def _context_lora(self, context, lora_A, lora_B, x, modality_scaling):
        if context is None or x.dim() < 3 or self.base_layer.in_features != self.hidden_size:
            return 0

        batch_size = x.size(0)
        current_context = self._align_context(context, batch_size)
        if current_context is None:
            return 0

        current_context = current_context.to(device=x.device, dtype=x.dtype).unsqueeze(1).expand(-1, x.size(1), -1)
        scale = self.scaling * modality_scaling.to(device=x.device, dtype=x.dtype)
        return self.lora_dropout(current_context) @ lora_A.transpose(0, 1).to(x.dtype) @ lora_B.transpose(0, 1).to(x.dtype) * scale

    def forward(self, x):
        lora_t = self.lora_dropout(x) @ self.lora_A_t.transpose(0, 1).to(x.dtype) @ self.lora_B_t.transpose(0, 1).to(x.dtype) * self.scaling
        lora_ui = self._context_lora(self.x_ui, self.lora_A_ui, self.lora_B_ui, x, self.ui_multimodal_scaling)
        lora_img = self._context_lora(self.x_img, self.lora_A_img, self.lora_B_img, x, self.image_multimodal_scaling)

        result = self.base_layer(x) + lora_t + lora_ui + lora_img
        return result


class MultiModalAdaLoraLayer(MultiModalLoraLayer):
    """Separate adaptive-rank SVD adapters for text, user-item and image."""

    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__(base_layer, hidden_size=hidden_size, dtype=dtype, **kwargs)
        self.branches = ['t']
        if base_layer.in_features == hidden_size:
            self.branches.extend(('ui', 'img'))

        for branch in ('t', 'ui', 'img'):
            lora_a = getattr(self, f'lora_A_{branch}')
            lora_b = getattr(self, f'lora_B_{branch}')
            if branch not in self.branches:
                lora_a.requires_grad_(False)
                lora_b.requires_grad_(False)
                continue
            nn.init.normal_(lora_a, std=0.02)
            nn.init.normal_(lora_b, std=0.02)
            setattr(self, f'lora_E_{branch}', nn.Parameter(
                torch.zeros(lora_a.size(0), 1, dtype=torch.float32)
            ))

    def _svd_update(self, inputs, branch):
        lora_a = getattr(self, f'lora_A_{branch}')
        lora_b = getattr(self, f'lora_B_{branch}')
        lora_e = getattr(self, f'lora_E_{branch}')
        return (
            self.lora_dropout(inputs)
            @ (lora_a.to(inputs.dtype) * lora_e.to(inputs.dtype)).transpose(0, 1)
            @ lora_b.transpose(0, 1).to(inputs.dtype)
            * self.scaling
        )

    def _context_update(self, context, branch, x, modality_scaling):
        if context is None or branch not in self.branches or x.dim() < 3:
            return 0
        current_context = self._align_context(context, x.size(0))
        if current_context is None:
            return 0
        current_context = current_context.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        return self._svd_update(current_context, branch) * modality_scaling.to(x.dtype)

    def forward(self, x):
        result = self.base_layer(x) + self._svd_update(x, 't')
        result = result + self._context_update(self.x_ui, 'ui', x, self.ui_multimodal_scaling)
        return result + self._context_update(self.x_img, 'img', x, self.image_multimodal_scaling)


class AdaLoraRankAllocator:
    """Share a scheduled rank budget across every active SVD adapter branch."""

    def __init__(self, model, init_r, target_r, total_step, tinit=0, tfinal=0,
                 delta_t=1, beta1=0.85, beta2=0.85):
        if not 0 < target_r < init_r:
            raise ValueError('AdaLoRA requires 0 < target_r < init_r.')
        if total_step <= 0 or tinit < 0 or tfinal < 0 or tinit + tfinal >= total_step:
            raise ValueError('AdaLoRA requires total_step > tinit + tfinal >= 0.')
        if delta_t <= 0:
            raise ValueError('AdaLoRA delta_t must be positive.')

        self.factors = []
        for layer in model.modules():
            if isinstance(layer, MultiModalAdaLoraLayer):
                branches = layer.branches
            elif isinstance(layer, AdaLoraLayer):
                branches = ('t',)
            else:
                continue
            for branch in branches:
                self.factors.append(tuple(getattr(layer, f'lora_{part}_{branch}')
                                          for part in ('A', 'B', 'E')))
        if not self.factors:
            raise ValueError('No AdaLoRA projection layers were found.')

        self.init_budget = init_r * len(self.factors)
        self.target_budget = target_r * len(self.factors)
        self.total_step = total_step
        self.tinit = tinit
        self.tfinal = tfinal
        self.delta_t = delta_t
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothed = [torch.zeros_like(e, dtype=torch.float32).flatten() for _, _, e in self.factors]
        self.uncertainty = [torch.zeros_like(e, dtype=torch.float32).flatten() for _, _, e in self.factors]
        self.final_masks = None

    @torch.no_grad()
    def update_and_allocate(self, step):
        if step > self.total_step:
            raise ValueError('AdaLoRA optimizer step exceeds total_step.')
        scores = []
        for index, (lora_a, lora_b, lora_e) in enumerate(self.factors):
            if any(parameter.grad is None for parameter in (lora_a, lora_b, lora_e)):
                importance = torch.zeros_like(self.smoothed[index])
            else:
                importance = (
                    (lora_a.float() * lora_a.grad.float()).abs().mean(dim=1)
                    + (lora_b.float() * lora_b.grad.float()).abs().mean(dim=0)
                    + (lora_e.float() * lora_e.grad.float()).abs().flatten()
                )
            self.smoothed[index].mul_(self.beta1).add_(importance, alpha=1 - self.beta1)
            self.uncertainty[index].mul_(self.beta2).add_(
                (importance - self.smoothed[index]).abs(), alpha=1 - self.beta2
            )
            scores.append(self.smoothed[index] * self.uncertainty[index])

        if step <= self.tinit:
            return
        final_step = self.total_step - self.tfinal
        if step >= final_step and self.final_masks is not None:
            masks = self.final_masks
        else:
            if step < final_step and step % self.delta_t:
                return
            progress = min(1.0, (step - self.tinit) / (final_step - self.tinit))
            budget = int((self.init_budget - self.target_budget) * (1 - progress) ** 3
                         + self.target_budget)
            flat_scores = torch.cat(scores)
            selected = flat_scores.topk(budget).indices
            flat_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            flat_mask[selected] = True
            masks = flat_mask.split([e.numel() for _, _, e in self.factors])
            if step >= final_step:
                self.final_masks = masks

        for (_, _, lora_e), mask in zip(self.factors, masks):
            lora_e.masked_fill_(~mask.view_as(lora_e), 0.0)


def adalora_orthogonal_regularization(model):
    """Regularize the active AdaLoRA A and B factors in full precision."""
    penalties = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        factor = parameter.float()
        if 'lora_A_' in name:
            covariance = factor @ factor.T
        elif 'lora_B_' in name:
            covariance = factor.T @ factor
        else:
            continue
        identity = torch.eye(covariance.size(0), device=covariance.device)
        penalties.append(torch.linalg.matrix_norm(covariance - identity))
    return torch.stack(penalties).mean() if penalties else 0.0


class SharedMultiModalLoraLayer(nn.Module):
    def __init__(
        self,
        base_layer,
        hidden_size=4096,
        dtype=torch.bfloat16,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.base_layer = base_layer

        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 24)
        lora_alpha = kwargs.pop("lora_alpha", r)
        lora_dropout = kwargs.pop("lora_dropout", 0.0)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Only ONE shared LoRA operator
        self.lora_A = nn.Parameter(
            torch.randn(r, in_features, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, dtype=dtype)
        )

        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout)

        self.x_ui = None
        self.x_img = None

    def _align_context(self, context, batch_size):
        if context is None:
            return None

        if context.size(0) == batch_size:
            return context

        if batch_size % context.size(0) == 0:
            repeat = batch_size // context.size(0)
            return context.repeat_interleave(repeat, dim=0)

        context = context[:batch_size]

        return context if context.size(0) == batch_size else None

    def _shared_lora(self, z):
        return (
            self.lora_dropout(z)
            @ self.lora_A.transpose(0, 1).to(z.dtype)
            @ self.lora_B.transpose(0, 1).to(z.dtype)
            * self.scaling
        )

    def _context_lora(self, context, x):
        if (
            context is None
            or x.dim() < 3
            or self.base_layer.in_features != self.hidden_size
        ):
            return 0

        batch_size = x.size(0)

        current_context = self._align_context(
            context, batch_size
        )

        if current_context is None:
            return 0

        current_context = (
            current_context
            .to(device=x.device, dtype=x.dtype)
            .unsqueeze(1)
            .expand(-1, x.size(1), -1)
        )

        return self._shared_lora(current_context)

    def forward(self, x):

        # Same ΔW applied to text
        lora_t = self._shared_lora(x)

        # Same ΔW applied to UI
        lora_ui = self._context_lora(self.x_ui, x)

        # Same ΔW applied to image
        lora_img = self._context_lora(self.x_img, x)

        return (
            self.base_layer(x)
            + lora_t
            + lora_ui
            + lora_img
        )

class MoDLoRA(nn.Module):
    use_dora = False
    use_adalora = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, k, r, mlp_size, lora_modules, dtype=torch.bfloat16, image_embeddings=None,
                        ui_multimodal_scale=1.0, image_multimodal_scale=1.0, is_shared=False, **kwargs):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            quantization_config=quantization_config,
            torch_dtype=dtype, 
            **kwargs
        )
        #base_model.gradient_checkpointing_enable()
        return cls(
            base_model, nuser, nitem, k, r, mlp_size, lora_modules,
            dtype, pretrained_model_name_or_path, image_embeddings=image_embeddings,
            ui_multimodal_scale=ui_multimodal_scale,
            image_multimodal_scale=image_multimodal_scale,
            is_shared=is_shared
        )
    
    def __init__(self, base_model, nuser, nitem, k, r, mlp_size, lora_modules, dtype=torch.bfloat16, pretrained_model_name_or_path="",
                 image_embeddings=None, ui_multimodal_scale=1.0, image_multimodal_scale=1.0, is_shared=False):
        super().__init__()
        self.model = base_model
        self.dtype = dtype
        
        self.user_emb = nn.Embedding(nuser, k, dtype=dtype)
        self.item_emb = nn.Embedding(nitem, k, dtype=dtype)
        emsize = self.model.config.hidden_size
        self.r = r
        self.k = k
        self.f_r = NeuMF_Predictor(emsize=k, hidden_size=mlp_size)  
        self.f_user = nn.Linear(k, k, dtype=dtype)
        self.f_item = nn.Linear(k, k, dtype=dtype)
        self.f_ui = nn.Linear(k * 2, emsize, dtype=dtype)
        if image_embeddings is None:
            raise ValueError("image_embeddings is required for MoDLoRA image LoRA. Generate it with DataLoader and CLIP first.")
        image_embeddings = self.build_image_embedding_table(image_embeddings, nitem)
        if image_embeddings.dim() != 2:
            raise ValueError("image_embeddings should be a 2-D tensor shaped as (nitem, image_dim)")
        self.image_dim = image_embeddings.size(1)
        self.register_buffer("image_embeddings", image_embeddings)
        self.f_img = nn.Linear(self.image_dim, emsize, dtype=dtype)

        # Initialize user/item embeddings
        initrange = 0.1
        self.user_emb.weight.data.uniform_(-initrange, initrange)
        self.item_emb.weight.data.uniform_(-initrange, initrange)
       
        # Dynamically set target_modules based on model type
        model_name_lower = pretrained_model_name_or_path.lower()
        if 'llama' in model_name_lower or 'qwen' in model_name_lower or 'mistral' in model_name_lower or 'gemma' in model_name_lower:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj"]

        if lora_modules < len(module_list):
            target_modules = module_list[:lora_modules]
        else:
            target_modules = module_list

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        for name, module in self.model.named_modules():
            is_target_layer = any(t_name in name for t_name in target_modules)
            is_valid_class = isinstance(module, nn.Linear) or module.__class__.__name__ in ['Conv1D', 'Linear']
            
            if is_target_layer and is_valid_class:
                base_layer = module
                
                for param in module.parameters():
                    param.requires_grad = False
                
                in_f = getattr(base_layer, "in_features", getattr(base_layer, "nx", None))
                out_f = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
                
                if not hasattr(base_layer, "in_features"):
                    base_layer.in_features = in_f
                if not hasattr(base_layer, "out_features"):
                    base_layer.out_features = out_f
                
                # Build text + single UI + single image adapter layer.
                # Keep the same data fusion and rating path for all MoD variants.
                if self.use_adalora:
                    AdapterLayer = MultiModalAdaLoraLayer
                else:
                    AdapterLayer = MultiModalDoRALayer if self.use_dora else MultiModalLoraLayer
                new_layer = AdapterLayer(
                    base_layer=base_layer,
                    dtype=self.dtype,
                    r=r,
                    lora_alpha=r, 
                    lora_dropout=0.1, 
                    k=k, 
                    hidden_size=self.model.config.hidden_size,
                    ui_multimodal_scale=ui_multimodal_scale,
                    image_multimodal_scale=image_multimodal_scale
                )
                if is_shared:
                    shared_r = 3 * r
                    new_layer = SharedMultiModalLoraLayer(
                        base_layer=base_layer,
                        dtype=self.dtype,
                        r=shared_r,
                        lora_alpha=shared_r,
                        lora_dropout=0.1,
                        hidden_size=self.model.config.hidden_size
                    )
                
                parts = name.rsplit('.', 1)
                if len(parts) == 1:
                    parent_module = self.model
                    child_name = parts[0]
                else:
                    parent_name, child_name = parts
                    try:
                        parent_module = self.model.get_submodule(parent_name)
                    except AttributeError:
                        parent_module = self.model.base_model.get_submodule(parent_name)

                setattr(parent_module, child_name, new_layer)

        # === 4. Activate Trainable Parameters ===
        for param in self.user_emb.parameters():
            param.requires_grad = True
        for param in self.item_emb.parameters():
            param.requires_grad = True
        for param in self.f_user.parameters(): 
            param.requires_grad = True
        for param in self.f_item.parameters(): 
            param.requires_grad = True
        for param in self.f_ui.parameters(): 
            param.requires_grad = True
        for param in self.f_img.parameters():
            param.requires_grad = True
        for param in self.f_r.parameters(): 
            param.requires_grad = True

        total_trainable_params = 0
        total_all_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_all_params += num_params
            if param.requires_grad:
                total_trainable_params += num_params

        trainable_ratio = (total_trainable_params / total_all_params) * 100 if total_all_params > 0 else 0
        
        print(f"\n--- Trainable Parameters Summary ({'LLM base'}) ---")
        print(f"Total parameters: {total_all_params:,}")
        adapter_name = "MoDAdaLoRA" if self.use_adalora else ("MoDDoRA" if self.use_dora else "MoDLoRA")
        print(f"Trainable parameters ({adapter_name} + Embeddings): {total_trainable_params:,}")
        print(f"Trainable ratio: {trainable_ratio:.2f}%")
        print(f"------------------------------------")

    # Add method to propagate user-item and image contexts to all custom LoRA layers
    def set_Q(self, x_ui=None, x_img=None):
        if x_ui is not None:
            x_ui = F.normalize(x_ui.to(torch.float32), p=2, dim=-1).to(self.dtype)
        if x_img is not None:
            x_img = F.normalize(x_img.to(torch.float32), p=2, dim=-1).to(self.dtype)
        for module in self.modules():
            if isinstance(module, MultiModalLoraLayer) or isinstance(module, SharedMultiModalLoraLayer):
                module.x_ui = x_ui
                module.x_img = x_img

    @staticmethod
    def to_image_tensor(value):
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.numel() == 0:
            return None
        while tensor.dim() > 1:
            tensor = tensor.mean(dim=0)
        return tensor.flatten().contiguous()

    @classmethod
    def build_image_embedding_table(cls, image_embeddings, nitem):
        if isinstance(image_embeddings, dict):
            for table_key in ['image_embeddings', 'image_embedding', 'item_embeddings', 'embeddings', 'features']:
                if table_key in image_embeddings and not isinstance(image_embeddings[table_key], (int, float, str)):
                    image_embeddings = image_embeddings[table_key]
                    break

        if isinstance(image_embeddings, dict):
            features = {}
            for item_idx, feature in image_embeddings.items():
                if not isinstance(item_idx, int) or item_idx < 0 or item_idx >= nitem:
                    continue
                tensor = cls.to_image_tensor(feature)
                if tensor is not None:
                    features[item_idx] = tensor
            if not features:
                raise ValueError("No valid item image embeddings found")
            first_feature = next(iter(features.values()))
            feature_dim = first_feature.numel()
            table = torch.zeros((nitem, feature_dim), dtype=torch.float32)
            for item_idx, feature in features.items():
                copy_dim = min(feature_dim, feature.numel())
                table[item_idx, :copy_dim] = feature[:copy_dim]
            return table

        table = torch.as_tensor(image_embeddings, dtype=torch.float32)
        if table.dim() > 2:
            table = table.view(table.size(0), -1)
        if table.dim() != 2:
            raise ValueError("image_embeddings should be a 2-D table or a dict keyed by item index")
        if table.size(0) != nitem:
            resized_rows = torch.zeros((nitem, table.size(1)), dtype=torch.float32)
            copy_rows = min(nitem, table.size(0))
            resized_rows[:copy_rows] = table[:copy_rows]
            table = resized_rows
        return table.contiguous()

    def get_image_features(self, item):
        return self.image_embeddings[item].to(device=item.device, dtype=self.dtype)

    def build_lora_context(self, user, item):
        user_emb, item_emb = self.user_emb(user), self.item_emb(item)
        Q_ui = torch.cat([user_emb, item_emb], dim=-1)
        x_ui = self.f_ui(Q_ui)

        image_emb = self.get_image_features(item)
        x_img = self.f_img(image_emb)
        return x_ui, x_img

    #  Wrapper methods for base model functionalities
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens):
        self.model.resize_token_embeddings(new_num_tokens)
    
    def generate(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        item = kwargs.pop('item', None)
        
        if user is not None and item is not None:
            x_ui, x_img = self.build_lora_context(user, item)
            self.set_Q(x_ui, x_img)
            
        return self.model.generate(*args, **kwargs)

    def get_load_balancing_loss(self):
        return torch.tensor(0.0, device=self.user_emb.weight.device)
    
    def forward(self, input_ids, attention_mask=None, user=None, item=None, text_lens=None, **kwargs):
        device = input_ids.device
        x_ui, x_img = self.build_lora_context(user, item)
        self.set_Q(x_ui, x_img)

        labels = torch.full_like(input_ids, -100, dtype=torch.int64).to(device)
        if text_lens is not None:
            for i, t_len in enumerate(text_lens):
                labels[i, -t_len:] = input_ids[i, -t_len:]  
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        predicted_rating = self.predict_rating(user, item)
        return outputs, predicted_rating
        
    def predict_rating(self, user, item, llm_state=None):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        p_u = self.f_user(user_emb)
        q_i = self.f_item(item_emb)
        
        predicted_rating = self.f_r(p_u, q_i)
        return predicted_rating


def _dequantize_linear_weight(base_layer):
    """
    Return base-layer weight as a normal floating-point tensor.

    Supports:
      1. torch.nn.Linear
      2. bitsandbytes Linear8bitLt / Int8Params
      3. bitsandbytes Linear4bit / Params4bit (also works if used later)

    The DoRA implementation itself does NOT require PEFT,
    but for reliable bitsandbytes dequantization we preferentially
    use PEFT's helper.
    """

    # ---------------------------------------------------------
    # Preferred route: PEFT helper.
    # It correctly handles bnb Int8Params / Params4bit.
    # ---------------------------------------------------------
    try:
        from peft.utils.integrations import dequantize_module_weight
        return dequantize_module_weight(base_layer)

    except (ImportError, AttributeError):
        pass

    # ---------------------------------------------------------
    # Fallback: recent Transformers also provides a bnb helper.
    # ---------------------------------------------------------
    weight = base_layer.weight
    weight_cls = weight.__class__.__name__

    if weight_cls in ("Int8Params", "Params4bit"):
        try:
            from transformers.integrations.bitsandbytes import (
                dequantize_bnb_weight,
            )

            state = getattr(base_layer, "state", None)
            return dequantize_bnb_weight(weight, state)

        except (ImportError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                "The base layer is bitsandbytes-quantized, but its "
                "weight could not be safely dequantized. "
                "Install/update `peft`, which provides "
                "`peft.utils.integrations.dequantize_module_weight`."
            ) from exc

    # Ordinary nn.Linear
    return weight


class DoRALayer(nn.Module):
    """
    Drop-in replacement for the original LoraLayer.

    For an ordinary Linear layer:

        V = W0 + s * B A

        W_DoRA = diag(m / ||V||_row) V

    and

        y = W_DoRA x + bias

    where:
        - A, B learn the directional update;
        - m is a trainable magnitude vector;
        - s = lora_alpha / r.

    Notes
    -----
    * Designed for the Qwen / Mistral / Gemma linear modules used
      in the current code.
    * Compatible with base layers loaded with load_in_8bit=True.
    * The frozen quantized base layer is still used for its normal
      forward pass.
    * Only the current base matrix is temporarily dequantized when
      the DoRA normalization factor is calculated.
    """

    def __init__(
        self,
        base_layer,
        hidden_size=4096,
        dtype=torch.bfloat16,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.base_layer = base_layer

        # Freeze pretrained / quantized base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.0)
        eps = kwargs.pop("dora_eps", 1e-6)

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.eps = eps

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.in_features = in_features
        self.out_features = out_features

        # Put newly created adapter parameters on the same device
        # as the quantized base layer.
        adapter_device = base_layer.weight.device

        # -----------------------------------------------------
        # LoRA directional component
        # -----------------------------------------------------
        self.lora_A_t = nn.Parameter(
            torch.empty(
                r,
                in_features,
                dtype=dtype,
                device=adapter_device,
            )
        )

        self.lora_B_t = nn.Parameter(
            torch.zeros(
                out_features,
                r,
                dtype=dtype,
                device=adapter_device,
            )
        )

        # Standard LoRA-style initialization:
        # A random, B zero -> initial delta W = 0.
        nn.init.kaiming_uniform_(
            self.lora_A_t,
            a=math.sqrt(5)
        )
        nn.init.zeros_(self.lora_B_t)

        self.lora_dropout = nn.Dropout(lora_dropout)

        # -----------------------------------------------------
        # DoRA magnitude component
        #
        # m_i is initialized as ||W0_i||_2.
        #
        # Keep magnitude in FP32:
        # it is tiny compared with A/B and improves numerical
        # stability when the base model is int8/bfloat16.
        # -----------------------------------------------------
        with torch.no_grad():
            base_weight = _dequantize_linear_weight(
                self.base_layer
            )

            # Qwen/Mistral/Gemma use ordinary Linear convention:
            # [out_features, in_features].
            #
            # Optional fallback for GPT-style Conv1D.
            if (
                base_weight.shape[0] == in_features
                and base_weight.shape[1] == out_features
                and base_weight.shape
                != (out_features, in_features)
            ):
                base_weight = base_weight.transpose(0, 1)

            if base_weight.shape != (
                out_features,
                in_features,
            ):
                raise ValueError(
                    "Unexpected base weight shape for DoRA: "
                    f"{tuple(base_weight.shape)}, expected "
                    f"({out_features}, {in_features})."
                )

            magnitude_init = torch.linalg.vector_norm(
                base_weight.float(),
                ord=2,
                dim=1,
            )

            magnitude_init = magnitude_init.clamp_min(
                self.eps
            )

        self.dora_magnitude = nn.Parameter(
            magnitude_init.to(
                device=adapter_device,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        # Release the temporary dense dequantized matrix.
        del base_weight
        del magnitude_init

    @torch.no_grad()
    def _compute_weight_norm(self):
        """
        Compute

            || W0 + scaling * B A ||_2

        row-wise.

        Importantly:
          * no gradient is propagated through the norm;
          * W0 is dequantized only temporarily;
          * no permanent FP16/FP32 copy of W0 is stored.

        Detaching the normalization denominator follows the
        practical DoRA formulation.
        """

        # Dequantize only the current frozen layer.
        base_weight = _dequantize_linear_weight(
            self.base_layer
        )

        # Standard Linear orientation: [out, in]
        if (
            base_weight.shape[0] == self.in_features
            and base_weight.shape[1] == self.out_features
            and base_weight.shape
            != (self.out_features, self.in_features)
        ):
            base_weight = base_weight.transpose(0, 1)

        # Norm computation in FP32 for stability.
        base_weight = base_weight.float()

        # delta_W = B A
        #
        # This is intentionally constructed under no_grad().
        # Gradients for A/B come from the low-rank forward below,
        # not from the DoRA normalization denominator.
        delta_weight = torch.matmul(
            self.lora_B_t.float(),
            self.lora_A_t.float(),
        )

        direction_weight = (
            base_weight
            + self.scaling * delta_weight
        )

        weight_norm = torch.linalg.vector_norm(
            direction_weight,
            ord=2,
            dim=1,
        )

        weight_norm = weight_norm.clamp_min(
            self.eps
        )

        return weight_norm

    def forward(self, x):
        """
        Efficient DoRA forward:

        base = W0 x + b
        low_rank = BA x

        g = m / ||W0 + s BA||

        result =
            base
            + (g - 1) * W0 x
            + g * s * BA x

        which is equivalent to

            g (W0 + s BA) x + b

        while keeping the quantized W0 forward intact.
        """

        # -----------------------------------------------------
        # 1. Quantized frozen base forward.
        #
        # With Linear8bitLt this still uses bitsandbytes.
        # -----------------------------------------------------
        base_result = self.base_layer(x)

        result_dtype = base_result.dtype

        # -----------------------------------------------------
        # 2. Low-rank directional update
        # -----------------------------------------------------
        adapter_x = self.lora_dropout(x)

        # Explicit dtype conversion is useful because hidden-state
        # dtype can differ from adapter dtype under mixed precision.
        adapter_x = adapter_x.to(
            self.lora_A_t.dtype
        )

        lora_result = (
            adapter_x
            @ self.lora_A_t.transpose(0, 1)
            @ self.lora_B_t.transpose(0, 1)
        )

        # -----------------------------------------------------
        # 3. DoRA direction norm
        #
        # The norm itself is detached / no_grad.
        # -----------------------------------------------------
        weight_norm = self._compute_weight_norm()

        magnitude_scale = (
            self.dora_magnitude / weight_norm
        )

        # Cast only after division.
        magnitude_scale = magnitude_scale.to(
            device=base_result.device,
            dtype=result_dtype,
        )

        # Broadcast:
        #   [out] -> [1, 1, out] for transformer hidden states
        #
        # Also works for [B, out].
        scale_shape = (
            [1] * (base_result.dim() - 1)
            + [self.out_features]
        )

        magnitude_scale = magnitude_scale.view(
            *scale_shape
        )

        # -----------------------------------------------------
        # 4. Bias should NOT be magnitude-scaled.
        #
        # base_result = W0 x + b
        # base_no_bias = W0 x
        # -----------------------------------------------------
        base_no_bias = base_result

        bias = getattr(
            self.base_layer,
            "bias",
            None,
        )

        if bias is not None:
            bias_view_shape = (
                [1] * (base_result.dim() - 1)
                + [self.out_features]
            )

            bias_for_output = bias.to(
                device=base_result.device,
                dtype=result_dtype,
            ).view(*bias_view_shape)

            base_no_bias = (
                base_result - bias_for_output
            )

        # -----------------------------------------------------
        # 5. DoRA result
        #
        # base + DoRA residual:
        #
        # base
        # + (g - 1) base_without_bias
        # + g s BAx
        # -----------------------------------------------------
        dora_residual = (
            (magnitude_scale - 1.0)
            * base_no_bias
        )

        dora_residual = (
            dora_residual
            + magnitude_scale
            * lora_result.to(result_dtype)
            * self.scaling
        )

        result = base_result + dora_residual

        return result


class MultiModalDoRALayer(MultiModalLoraLayer):
    """DoRA magnitude decomposition for the three MoDLoRA branches.

    MoDLoRA can be viewed as a linear map over three separate inputs::

        y = W0 x + dW_text x + dW_ui q_ui + dW_img q_img + bias

    MoDDoRA treats the corresponding augmented matrix
    ``[W0 + dW_text, dW_ui, dW_img]`` as one direction and learns one
    output-channel magnitude vector. This preserves MoDLoRA's separate
    modality adapters while applying DoRA consistently to their joint output.
    """

    def __init__(
        self,
        base_layer,
        hidden_size=4096,
        dtype=torch.bfloat16,
        **kwargs
    ):
        eps = kwargs.pop("dora_eps", 1e-6)
        super().__init__(
            base_layer=base_layer,
            hidden_size=hidden_size,
            dtype=dtype,
            **kwargs
        )

        self.eps = eps
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # All B matrices start at zero, so initializing m with ||W0|| makes
        # the layer functionally identical to the pretrained base layer.
        with torch.no_grad():
            base_weight = _dequantize_linear_weight(self.base_layer)
            if (
                base_weight.shape[0] == self.in_features
                and base_weight.shape[1] == self.out_features
                and base_weight.shape != (self.out_features, self.in_features)
            ):
                base_weight = base_weight.transpose(0, 1)

            if base_weight.shape != (self.out_features, self.in_features):
                raise ValueError(
                    "Unexpected base weight shape for MoDDoRA: "
                    f"{tuple(base_weight.shape)}, expected "
                    f"({self.out_features}, {self.in_features})."
                )

            magnitude_init = torch.linalg.vector_norm(
                base_weight.float(), ord=2, dim=1
            ).clamp_min(self.eps)

        self.dora_magnitude = nn.Parameter(
            magnitude_init.to(
                device=base_layer.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        del base_weight
        del magnitude_init

    @torch.no_grad()
    def _compute_weight_norm(self):
        """Return row norms of the augmented multimodal direction."""
        base_weight = _dequantize_linear_weight(self.base_layer)
        if (
            base_weight.shape[0] == self.in_features
            and base_weight.shape[1] == self.out_features
            and base_weight.shape != (self.out_features, self.in_features)
        ):
            base_weight = base_weight.transpose(0, 1)

        base_direction = (
            base_weight.float()
            + self.scaling
            * torch.matmul(self.lora_B_t.float(), self.lora_A_t.float())
        )
        squared_norm = base_direction.square().sum(dim=1)

        # UI/image contexts are hidden-size inputs. For projections whose
        # input width differs from hidden_size, these branches are inactive.
        if self.in_features == self.hidden_size:
            ui_direction = (
                self.scaling
                * self.ui_multimodal_scaling.detach().float()
                * torch.matmul(self.lora_B_ui.float(), self.lora_A_ui.float())
            )
            image_direction = (
                self.scaling
                * self.image_multimodal_scaling.detach().float()
                * torch.matmul(self.lora_B_img.float(), self.lora_A_img.float())
            )
            squared_norm = (
                squared_norm
                + ui_direction.square().sum(dim=1)
                + image_direction.square().sum(dim=1)
            )

        return torch.sqrt(squared_norm.clamp_min(self.eps ** 2))

    def forward(self, x):
        base_result = self.base_layer(x)
        result_dtype = base_result.dtype

        lora_t = (
            self.lora_dropout(x)
            @ self.lora_A_t.transpose(0, 1).to(x.dtype)
            @ self.lora_B_t.transpose(0, 1).to(x.dtype)
            * self.scaling
        )
        lora_ui = self._context_lora(
            self.x_ui,
            self.lora_A_ui,
            self.lora_B_ui,
            x,
            self.ui_multimodal_scaling,
        )
        lora_img = self._context_lora(
            self.x_img,
            self.lora_A_img,
            self.lora_B_img,
            x,
            self.image_multimodal_scaling,
        )

        weight_norm = self._compute_weight_norm()
        magnitude_scale = (self.dora_magnitude / weight_norm).to(
            device=base_result.device,
            dtype=result_dtype,
        )
        scale_shape = [1] * (base_result.dim() - 1) + [self.out_features]
        magnitude_scale = magnitude_scale.view(*scale_shape)

        base_no_bias = base_result
        bias = getattr(self.base_layer, "bias", None)
        if bias is not None:
            bias_shape = [1] * (base_result.dim() - 1) + [self.out_features]
            base_no_bias = base_result - bias.to(
                device=base_result.device,
                dtype=result_dtype,
            ).view(*bias_shape)

        adapter_result = lora_t + lora_ui + lora_img
        return (
            base_result
            + (magnitude_scale - 1.0) * base_no_bias
            + magnitude_scale * adapter_result.to(result_dtype)
        )


class MoDDoRA(MoDLoRA):
    """MoDLoRA pipeline using multimodal DoRA adapter layers."""

    use_dora = True


class MoDAdaLoRA(MoDLoRA):
    """MoDLoRA pipeline with a shared adaptive rank budget."""

    use_adalora = True


class Concat_LoRA(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, k, r, mlp_size, lora_modules,
                        dtype=torch.bfloat16, image_embeddings=None, is_dora=False, is_adalora=False, **kwargs):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            quantization_config=quantization_config,
            torch_dtype=dtype, 
            **kwargs
        )
        #base_model.gradient_checkpointing_enable()
        return cls(
            base_model, nuser, nitem, k, r, mlp_size, lora_modules, dtype,
            pretrained_model_name_or_path, image_embeddings=image_embeddings,
            is_dora=is_dora, is_adalora=is_adalora
        )

    def __init__(self, base_model, nuser, nitem, k, r, mlp_size, lora_modules, dtype,
                 pretrained_model_name_or_path, image_embeddings=None, is_dora=False,
                 is_adalora=False):
        super().__init__()
        self.model = base_model
        self.dtype = dtype
        self.r = r
        self.k = k
        
        # Recommendation Embeddings
        self.user_emb = nn.Embedding(nuser, k, dtype=dtype)
        self.item_emb = nn.Embedding(nitem, k, dtype=dtype)
        self.hidden_size = self.model.config.hidden_size
        
        # Map lays for recommendation embeddings
        self.user_projector = nn.Linear(k, self.hidden_size, dtype=dtype)
        self.item_projector = nn.Linear(k, self.hidden_size, dtype=dtype)
        if image_embeddings is None:
            raise ValueError("image_embeddings is required for multimodal Concat_LoRA. Generate it with DataLoader and CLIP first.")
        image_embeddings = MoDLoRA.build_image_embedding_table(image_embeddings, nitem)
        if image_embeddings.dim() != 2:
            raise ValueError("image_embeddings should be a 2-D tensor shaped as (nitem, image_dim)")
        self.image_dim = image_embeddings.size(1)
        self.register_buffer("image_embeddings", image_embeddings)
        self.image_projector = nn.Linear(self.image_dim, self.hidden_size, dtype=dtype)
        
        # Rating prediction for user/item embeddings
        self.f_r = NeuMF_Predictor(emsize=k, hidden_size=mlp_size)
        self.f_user = nn.Linear(k, k, dtype=dtype)
        self.f_item = nn.Linear(k, k, dtype=dtype)

        # Initialize weights
        initrange = 0.1
        self.user_emb.weight.data.uniform_(-initrange, initrange)
        self.item_emb.weight.data.uniform_(-initrange, initrange)

        # Trainable Layers of adapter
        model_name_lower = pretrained_model_name_or_path.lower()
        if 'llama' in model_name_lower or 'qwen' in model_name_lower or 'mistral' in model_name_lower or 'gemma' in model_name_lower:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj"]

        target_modules = module_list[:lora_modules] if lora_modules < len(module_list) else module_list

        # Fraze LLM weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Build LoRA (LoraLayer)
        for name, module in self.model.named_modules():
            is_target_layer = any(t_name in name for t_name in target_modules)
            is_valid_class = isinstance(module, nn.Linear) or module.__class__.__name__ in ['Conv1D', 'Linear']
            
            if is_target_layer and is_valid_class:
                base_layer = module
                
                in_f = getattr(base_layer, "in_features", getattr(base_layer, "nx", None))
                out_f = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
                if not hasattr(base_layer, "in_features"): base_layer.in_features = in_f
                if not hasattr(base_layer, "out_features"): base_layer.out_features = out_f

                if is_adalora:
                    AdapterLayer = AdaLoraLayer
                else:
                    AdapterLayer = DoRALayer if is_dora else LoraLayer
                new_layer = AdapterLayer(
                    base_layer=base_layer,
                    dtype=self.dtype,
                    r=r,
                    lora_alpha=r,
                    lora_dropout=0.1,
                    hidden_size=self.hidden_size
                )
                                    
                parts = name.rsplit('.', 1)
                parent_module = self.model.get_submodule(parts[0]) if len(parts) > 1 else self.model
                setattr(parent_module, parts[-1], new_layer)

        # Activate trainable parameters
        for param in self.user_emb.parameters(): param.requires_grad = True
        for param in self.item_emb.parameters(): param.requires_grad = True
        for param in self.user_projector.parameters(): param.requires_grad = True
        for param in self.item_projector.parameters(): param.requires_grad = True
        for param in self.image_projector.parameters(): param.requires_grad = True
        for param in self.f_user.parameters(): param.requires_grad = True
        for param in self.f_item.parameters(): param.requires_grad = True
        for param in self.f_r.parameters(): param.requires_grad = True

        total_trainable_params = 0
        total_all_params = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_all_params += num_params
            if param.requires_grad:
                total_trainable_params += num_params

        trainable_ratio = (total_trainable_params / total_all_params) * 100 if total_all_params > 0 else 0
        
        print(f"\n--- Trainable Parameters Summary ({'LLM base'}) ---")
        print(f"Total parameters: {total_all_params:,}")
        print(f"Trainable parameters (LoRA + Embeddings): {total_trainable_params:,}")
        print(f"Trainable ratio: {trainable_ratio:.2f}%")
        print(f"------------------------------------")

    def resize_token_embeddings(self, new_num_tokens):
        self.model.resize_token_embeddings(new_num_tokens)

    def get_image_features(self, item):
        return self.image_embeddings[item].to(device=item.device, dtype=self.dtype)
    
    def _get_concat_embeddings(self, user, item, input_ids, attention_mask):
        """
        Left Padding Batchify, adding [User_Token, Item_Token, Image_Token] inserted after [Pad] and before the valid text:
        [Pad...Pad] + [u] + [i] + [x_img] + [Prompt] + <bos> + ...
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        hidden_size = self.hidden_size
        prefix_len = 3

        # 1. Extract Embeddings (B, S, H)
        text_tokens = self.model.get_input_embeddings()(input_ids.to(device))

        # 2. User-item-image tokens: (B, 3, H)
        u_emb = self.user_projector(self.user_emb(user)).unsqueeze(1)  
        i_emb = self.item_projector(self.item_emb(item)).unsqueeze(1)  
        img_emb = self.image_projector(self.get_image_features(item)).unsqueeze(1)
        ui_tokens = torch.cat([u_emb, i_emb, img_emb], dim=1)

        # 3. Reconstruct Embedding and mask: (B, 3 + S, H)
        full_embeds = torch.zeros((batch_size, prefix_len + seq_len, hidden_size), dtype=self.dtype, device=device)
        full_mask = torch.zeros((batch_size, prefix_len + seq_len), dtype=attention_mask.dtype, device=device)
        for idx in range(batch_size):
            # locate the position of the first 1 in the mask: the exact number of pads on the left side.
            ones_indices = (attention_mask[idx] == 1).nonzero(as_tuple=True)[0]
            num_pads = int(ones_indices[0].item()) if len(ones_indices) > 0 else 0

            # keep padding
            if num_pads > 0:
                full_embeds[idx, :num_pads] = text_tokens[idx, :num_pads]
                
            # insert [User, Item, Image]
            full_embeds[idx, num_pads : num_pads + prefix_len] = ui_tokens[idx]
            full_mask[idx, num_pads : num_pads + prefix_len] = 1
            
            # move valid texts: (Prompt + Review or Prompt)
            full_embeds[idx, num_pads + prefix_len :] = text_tokens[idx, num_pads:]
            full_mask[idx, num_pads + prefix_len :] = attention_mask[idx, num_pads:]

        return full_embeds, full_mask

    def forward(self, input_ids, attention_mask=None, user=None, item=None, text_lens=None, **kwargs):
        device = input_ids.device
        
        full_embeds, full_mask = self._get_concat_embeddings(user, item, input_ids, attention_mask)
        labels = torch.full((full_embeds.shape[0], full_embeds.shape[1]), -100, dtype=torch.int64, device=device)
        if text_lens is not None:
            for i, t_len in enumerate(text_lens):
                start_idx_in_input = input_ids.shape[1] - t_len
                start_idx_in_labels = start_idx_in_input + 3
                labels[i, start_idx_in_labels:] = input_ids[i, start_idx_in_input:]

        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        predicted_rating = self.predict_rating(user, item)
        return outputs, predicted_rating

    def generate(self, *args, **kwargs):
        input_ids = None
        attention_mask = None
        
        args_list = list(args)
        if len(args_list) > 0:
            input_ids = args_list.pop(0)
        if len(args_list) > 0:
            attention_mask = args_list.pop(0)
        
        if input_ids is None: input_ids = kwargs.pop('input_ids', None)
        if attention_mask is None: attention_mask = kwargs.pop('attention_mask', None)
        user = kwargs.pop('user', None)
        item = kwargs.pop('item', None)

        full_embeds, full_mask = self._get_concat_embeddings(user, item, input_ids, attention_mask)
        kwargs['inputs_embeds'] = full_embeds
        kwargs['attention_mask'] = full_mask
        
        return self.model.generate(*tuple(args_list), **kwargs)

    def predict_rating(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        p_u = self.f_user(user_emb)
        q_i = self.f_item(item_emb)
        rating = self.f_r(p_u, q_i)
        return rating
