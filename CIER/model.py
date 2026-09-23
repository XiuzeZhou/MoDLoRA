import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PromptEncoder(nn.Module):
    def __init__(self, user_num, item_num, tokenizer, hidden=1024, output_hidden=4096, use_image_embedding=False):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.use_image_embedding = use_image_embedding
        self.dropout = nn.Dropout(0.1)
        self.user_embedding = nn.Embedding(user_num, hidden)
        self.item_embedding = nn.Embedding(item_num, hidden)
        self.mlp_u = nn.Sequential(
            torch.nn.Linear(hidden, output_hidden)
        )
        self.mlp_v = nn.Sequential(
            torch.nn.Linear(hidden, output_hidden)
        )
        # Replace original self.instruction and self.verbalizer
        def safe_tokenize(text):
            tokens = tokenizer(text)['input_ids']
            # If the tokenizer automatically adds bos or cls, safely remove the first token
            if len(tokens) > 0 and tokens[0] in [tokenizer.bos_token_id, tokenizer.cls_token_id]:
                return tokens[1:]
            return tokens

        self.instruction = torch.tensor([safe_tokenize('Predict the rating for the given user and item, and generate a corresponding explanation or keyword.')])
        self.hard_prompt1 = torch.tensor([safe_tokenize('The rating given by user')])
        self.hard_prompt2 = torch.tensor([safe_tokenize('to item')])
        self.hard_prompt3 = torch.tensor([safe_tokenize('is ')])
        self.hard_prompt4 = torch.tensor([safe_tokenize('and the corresponding')])
        self.hard_prompt5 = torch.tensor([safe_tokenize('is "')])
        
        self.sub_full_words = torch.tensor(safe_tokenize('keyword explanation'))
        
        self.verbalizer = [tokenizer(str(i), add_special_tokens=False)['input_ids'][-1] for i in range(1, 6)]
        image_slot = 1 if self.use_image_embedding else 0
        self.prompt_length = 4 + image_slot + self.instruction.shape[1] + self.hard_prompt1.shape[1] + self.hard_prompt2.shape[1] + self.hard_prompt3.shape[1] + self.hard_prompt4.shape[1] + self.hard_prompt5.shape[1]
        self.rating_index = 1 + image_slot + self.instruction.shape[1] + self.hard_prompt1.shape[1] + self.hard_prompt2.shape[1] + self.hard_prompt3.shape[1]
    
    def forward(self, user_id=None, item_id=None,
                rating=None,embed_tokens=None,curr_flag=None,image_embedding=None
               ):
        device = user_id.device
        user_embedding = self.mlp_u(self.user_embedding(user_id)).unsqueeze(1)
        item_embedding = self.mlp_v(self.item_embedding(item_id)).unsqueeze(1)
        if self.use_image_embedding:
            if image_embedding is None:
                raise ValueError("image_embedding is required when PromptEncoder uses image embeddings.")
            if image_embedding.dim() == 2:
                image_embedding = image_embedding.unsqueeze(1)
        if rating is None:
            prompt_parts = [
                embed_tokens(self.instruction.to(device)).repeat(user_embedding.shape[0],1,1),
                embed_tokens(self.hard_prompt1.to(device)).repeat(user_embedding.shape[0],1,1),
                user_embedding,
                embed_tokens(self.hard_prompt2.to(device)).repeat(user_embedding.shape[0],1,1),
                item_embedding,
            ]
            if self.use_image_embedding:
                prompt_parts.append(image_embedding)
            prompt_parts.append(embed_tokens(self.hard_prompt3.to(device)).repeat(user_embedding.shape[0],1,1))
            return torch.cat(prompt_parts, dim=-2)
        values = embed_tokens(torch.tensor(self.verbalizer).to(device)).unsqueeze(0).repeat(rating.shape[0],1,1)
        values = (rating.unsqueeze(-1) * values).sum(dim=1)
        if curr_flag is None:
            flag_words = self.sub_full_words.to(device)[[1]].repeat(user_embedding.shape[0],1)
        else:
            flag_words = self.sub_full_words.to(device)[curr_flag].unsqueeze(1)
        prompt_parts = [
            embed_tokens(self.instruction.to(device)).repeat(user_embedding.shape[0],1,1),
            embed_tokens(self.hard_prompt1.to(device)).repeat(user_embedding.shape[0],1,1),
            user_embedding,
            embed_tokens(self.hard_prompt2.to(device)).repeat(user_embedding.shape[0],1,1),
            item_embedding,
        ]
        if self.use_image_embedding:
            prompt_parts.append(image_embedding)
        prompt_parts.extend([
            embed_tokens(self.hard_prompt3.to(device)).repeat(user_embedding.shape[0],1,1),
            values.unsqueeze(1),
            embed_tokens(self.hard_prompt4.to(device)).repeat(user_embedding.shape[0],1,1),
            embed_tokens(flag_words),
            embed_tokens(self.hard_prompt5.to(device)).repeat(user_embedding.shape[0],1,1),
        ])
        return torch.cat(prompt_parts, dim=-2)

class MyModel(nn.Module):
    def __init__(self, user_num, item_num,  hidden, llm_hidden, tokenizer, image_embeddings=None):
        super(MyModel, self).__init__()
        self.use_prompt_image = image_embeddings is not None
        self.prompt_encoder = PromptEncoder(user_num, item_num, tokenizer, hidden=hidden, output_hidden=llm_hidden, use_image_embedding=self.use_prompt_image)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.dropout = nn.Dropout(0.1)
        if self.use_prompt_image:
            image_embeddings = self.build_image_embedding_table(image_embeddings, item_num)
            if image_embeddings.dim() != 2:
                raise ValueError("image_embeddings should be a 2-D tensor shaped as (item_num, image_dim)")
            self.prompt_image_dim = image_embeddings.size(1)
            self.register_buffer("prompt_image_embeddings", image_embeddings)
            self.f_prompt_img = nn.Linear(self.prompt_image_dim, llm_hidden)
        else:
            self.prompt_image_dim = None
            self.f_prompt_img = None
        self.reset_parameters()
        self.model = None

        self.generate_weight = 1.0
        self.rating_weight= 0.1
        
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def to_image_tensor(value):
        if value is None:
            return None
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            return None
        if tensor.numel() == 0:
            return None
        while tensor.dim() > 1:
            tensor = tensor.mean(dim=0)
        return tensor.flatten().contiguous()

    @classmethod
    def build_image_embedding_table(cls, image_embeddings, item_num):
        if isinstance(image_embeddings, dict):
            for table_key in ['image_embeddings', 'image_embedding', 'item_embeddings', 'embeddings', 'features']:
                if table_key in image_embeddings and not isinstance(image_embeddings[table_key], (int, float, str)):
                    image_embeddings = image_embeddings[table_key]
                    break

        if isinstance(image_embeddings, dict):
            features = {}
            for item_idx, feature in image_embeddings.items():
                if not isinstance(item_idx, int) or item_idx < 0 or item_idx >= item_num:
                    continue
                tensor = cls.to_image_tensor(feature)
                if tensor is not None:
                    features[item_idx] = tensor
            if not features:
                raise ValueError("No valid item image embeddings found")
            first_feature = next(iter(features.values()))
            feature_dim = first_feature.numel()
            table = torch.zeros((item_num, feature_dim), dtype=torch.float32)
            for item_idx, feature in features.items():
                copy_dim = min(feature_dim, feature.numel())
                table[item_idx, :copy_dim] = feature[:copy_dim]
            return table.contiguous()

        table = torch.as_tensor(image_embeddings, dtype=torch.float32)
        if table.dim() > 2:
            table = table.view(table.size(0), -1)
        if table.dim() != 2:
            raise ValueError("image_embeddings should be a 2-D table or a dict keyed by item index")
        if table.size(0) != item_num:
            resized_rows = torch.zeros((item_num, table.size(1)), dtype=torch.float32)
            copy_rows = min(item_num, table.size(0))
            resized_rows[:copy_rows] = table[:copy_rows]
            table = resized_rows
        return table.contiguous()

    def get_prompt_image_embedding(self, item_id):
        image_features = self.prompt_image_embeddings[item_id].to(
            device=item_id.device,
            dtype=self.f_prompt_img.weight.dtype
        )
        return self.f_prompt_img(image_features)
                
    def get_embedding(self, input_ids=None,  user_id=None, item_id=None, rating=None,curr_flag=None):
        if hasattr(self.model, "get_input_embeddings"):
            embeddings = self.model.get_input_embeddings()
        else:
            # Compatible with PEFT
            embeddings = self.model.base_model.get_input_embeddings()
        
        image_embedding = self.get_prompt_image_embedding(item_id) if self.use_prompt_image else None
        if input_ids is None:
            return self.prompt_encoder(user_id=user_id,item_id=item_id,embed_tokens=embeddings,image_embedding=image_embedding)
        prompt = self.prompt_encoder(user_id=user_id,item_id=item_id,
                                     rating=rating,embed_tokens=embeddings,curr_flag=curr_flag,image_embedding=image_embedding
                                    )
        if input_ids.shape[1]==0:
            return prompt
        inputs_embeds = embeddings(input_ids)
        inputs_embeds = torch.cat([prompt,inputs_embeds],dim=-2)
        return inputs_embeds
    
    def forward(self, input_ids=None, user_id=None, item_id=None,
                rating=None, kv_cache=None
               ):
        #enocde
        if kv_cache == None:
            inputs_embeds = self.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id, rating=rating)
            output = self.model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
        else:
            output = self.model(input_ids=input_ids, past_key_values=kv_cache, use_cache=True, return_dict=True)
        #decode
        kv_cache = getattr(output, "past_key_values", None)
        if kv_cache is None:
            kv_cache = output.get("past_key_values", None)
        if kv_cache is None:
            raise RuntimeError("Model output does not contain past_key_values. Enable use_cache before generation.")
        logits = output['logits'][:,-1,:]
        logits = torch.softmax(logits,dim=1)
        
        return logits, kv_cache
    def rating_predict(self, user_id=None, item_id=None):
        
        inputs_embeds = self.get_embedding(user_id=user_id, item_id=item_id)
        logits = self.model(inputs_embeds=inputs_embeds)['logits']
        #decode
        output = logits[:,self.prompt_encoder.rating_index,:]
        output = output[:,self.prompt_encoder.verbalizer]
        output = torch.softmax(output,dim=1)
        
        return output
    def train_step(self, input_ids, user_id=None, item_id=None, rating=None, curr_flag=None,
                   rating_input=None
                  ):
        #enocde
        inputs_embeds = self.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id,
                                              rating=rating_input,curr_flag=curr_flag
                                             )
        logits = self.model(inputs_embeds=inputs_embeds)['logits']
        output = logits[:,self.prompt_encoder.rating_index,:]
        output = output[:,self.prompt_encoder.verbalizer]
        loss = F.cross_entropy(output,rating)*self.rating_weight
        
        #MLM
        logits = logits[:,self.prompt_encoder.prompt_length-1:-1,:]
        targets = input_ids
        y_mask = input_ids.clone()
        y_mask[targets!=0] = 1
        y_mask = y_mask.reshape(-1)
        targets = targets.reshape(-1)
        logits = logits.reshape(-1,logits.shape[-1])
        generate_loss = (self.ce_loss(logits,targets) * y_mask).sum(dim=0) / (y_mask.sum(dim=0))
        loss += self.generate_weight*generate_loss

            
        return loss


# =========================================================================
# MoDLoRA multimodal LoRA framework
# =========================================================================
class MultiModalLoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 8)
        lora_alpha = kwargs.pop("lora_alpha", 32)
        lora_dropout = kwargs.pop("lora_dropout", 0.05)
        self.use_image_lora = kwargs.pop("use_image_lora", False)
        ui_multimodal_scale = kwargs.pop("ui_multimodal_scale", 1.0)
        image_multimodal_scale = kwargs.pop("image_multimodal_scale", 1.0)
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A_t = nn.Parameter(torch.empty(r, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A_t, a=math.sqrt(5))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))

        self.lora_A_ui = nn.Parameter(torch.empty(r, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A_ui, a=math.sqrt(5))
        self.lora_B_ui = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        self.ui_multimodal_scaling = nn.Parameter(torch.tensor(float(ui_multimodal_scale), dtype=torch.float32))

        if self.use_image_lora:
            self.lora_A_img = nn.Parameter(torch.empty(r, in_features, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A_img, a=math.sqrt(5))
            self.lora_B_img = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
            self.image_multimodal_scaling = nn.Parameter(torch.tensor(float(image_multimodal_scale), dtype=torch.float32))
        else:
            self.lora_A_img = None
            self.lora_B_img = None
            self.image_multimodal_scaling = None

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
            return context.repeat_interleave(batch_size // context.size(0), dim=0)
        context = context[:batch_size]
        return context if context.size(0) == batch_size else None

    def _context_lora(self, context, lora_A, lora_B, x, modality_scaling):
        if context is None or self.base_layer.in_features != self.hidden_size:
            return 0

        current_context = self._align_context(context, x.size(0))
        if current_context is None:
            return 0

        current_context = current_context.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        scale = self.scaling * modality_scaling.to(device=x.device, dtype=x.dtype)
        return self.lora_dropout(current_context) @ lora_A.transpose(0, 1).to(x.dtype) \
                                        @ lora_B.transpose(0, 1).to(x.dtype) * scale

    def forward(self, x):
        lora_t = self.lora_dropout(x) @ self.lora_A_t.transpose(0, 1).to(x.dtype) \
                                @ self.lora_B_t.transpose(0, 1).to(x.dtype) * self.scaling
        lora_ui = self._context_lora(self.x_ui, self.lora_A_ui, self.lora_B_ui, x, self.ui_multimodal_scaling)
        lora_img = 0
        if self.use_image_lora:
            lora_img = self._context_lora(self.x_img, self.lora_A_img, self.lora_B_img, x, self.image_multimodal_scaling)

        return self.base_layer(x) + lora_t + lora_ui + lora_img


class MoDLoRA(MyModel):
    def __init__(self, user_num, item_num, hidden, llm_hidden, tokenizer, model_llm, r=8, lora_alpha=32, lora_modules=2, image_embeddings=None, ui_multimodal_scale=1.0, image_multimodal_scale=1.0):
        super().__init__(user_num, item_num, hidden, llm_hidden, tokenizer)
        self.model = model_llm  # base LLM
        
        self.f_ui = nn.Linear(hidden * 2, llm_hidden, dtype=model_llm.dtype)
        self.use_image_lora = image_embeddings is not None
        if self.use_image_lora:
            image_embeddings = self.build_image_embedding_table(image_embeddings, item_num)
            if image_embeddings.dim() != 2:
                raise ValueError("image_embeddings should be a 2-D tensor shaped as (item_num, image_dim)")
            self.image_dim = image_embeddings.size(1)
            self.register_buffer("image_embeddings", image_embeddings)
            self.f_img = nn.Linear(self.image_dim, llm_hidden, dtype=model_llm.dtype)
        else:
            self.image_dim = None
            self.f_img = None
        
        # Dynamically set target_modules based on model type
        module_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = module_list[:lora_modules]  # ["q_proj", "k_proj"] #["q_proj", "v_proj", "k_proj", "o_proj"]
        for param in self.model.parameters():
            param.requires_grad = False
            
        for name, module in self.model.named_modules():
            if any(t_name in name for t_name in target_modules) and isinstance(module, nn.Linear):
                new_layer = MultiModalLoraLayer(
                    base_layer=module, dtype=model_llm.dtype, r=r, lora_alpha=lora_alpha, 
                    hidden_size=llm_hidden,
                    use_image_lora=self.use_image_lora,
                    ui_multimodal_scale=ui_multimodal_scale,
                    image_multimodal_scale=image_multimodal_scale
                )
                parts = name.rsplit('.', 1)
                parent = self.model.get_submodule(parts[0]) if len(parts) > 1 else self.model
                setattr(parent, parts[-1], new_layer)

    @staticmethod
    def to_image_tensor(value):
        if value is None:
            return None
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            return None
        if tensor.numel() == 0:
            return None
        while tensor.dim() > 1:
            tensor = tensor.mean(dim=0)
        return tensor.flatten().contiguous()

    @classmethod
    def build_image_embedding_table(cls, image_embeddings, item_num):
        if isinstance(image_embeddings, dict):
            for table_key in ['image_embeddings', 'image_embedding', 'item_embeddings', 'embeddings', 'features']:
                if table_key in image_embeddings and not isinstance(image_embeddings[table_key], (int, float, str)):
                    image_embeddings = image_embeddings[table_key]
                    break

        if isinstance(image_embeddings, dict):
            features = {}
            for item_idx, feature in image_embeddings.items():
                if not isinstance(item_idx, int) or item_idx < 0 or item_idx >= item_num:
                    continue
                tensor = cls.to_image_tensor(feature)
                if tensor is not None:
                    features[item_idx] = tensor
            if not features:
                raise ValueError("No valid item image embeddings found")
            first_feature = next(iter(features.values()))
            feature_dim = first_feature.numel()
            table = torch.zeros((item_num, feature_dim), dtype=torch.float32)
            for item_idx, feature in features.items():
                copy_dim = min(feature_dim, feature.numel())
                table[item_idx, :copy_dim] = feature[:copy_dim]
            return table.contiguous()

        table = torch.as_tensor(image_embeddings, dtype=torch.float32)
        if table.dim() > 2:
            table = table.view(table.size(0), -1)
        if table.dim() != 2:
            raise ValueError("image_embeddings should be a 2-D table or a dict keyed by item index")
        if table.size(0) != item_num:
            resized_rows = torch.zeros((item_num, table.size(1)), dtype=torch.float32)
            copy_rows = min(item_num, table.size(0))
            resized_rows[:copy_rows] = table[:copy_rows]
            table = resized_rows
        return table.contiguous()

    def get_image_features(self, item_id):
        return self.image_embeddings[item_id].to(device=item_id.device, dtype=self.model.dtype)

    def get_embedding(self, input_ids=None, user_id=None, item_id=None, rating=None, curr_flag=None):
        u_emb = self.prompt_encoder.user_embedding(user_id)
        i_emb = self.prompt_encoder.item_embedding(item_id)
        x_ui = self.f_ui(torch.cat([u_emb, i_emb], dim=-1))
        x_img = self.f_img(self.get_image_features(item_id)) if self.use_image_lora else None
        
        for mod in self.modules():
            if isinstance(mod, MultiModalLoraLayer):
                mod.x_ui = x_ui
                mod.x_img = x_img
        
        return super().get_embedding(input_ids, user_id, item_id, rating, curr_flag)
