import re
import torch
import torch.nn as nn
import logging
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


class MultiPromptEOLEmbeddingsLLM(nn.Module):
    def __init__(self, model, layer='32', tokenizer=None, max_length=1024, short_template='', long_template=''):
        super(MultiPromptEOLEmbeddingsLLM, self).__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self.embedding_layers = layer
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer_name = tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            # if 'llama' in tokenizer:
            #     tokenizer = LlamaTokenizer.from_pretrained(tokenizer, use_fast=True)
            #     tokenizer.bos_token_id = 1
            #     tokenizer.eos_token = '</s>'
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained(tokenizer)

            self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
            self.tokenizer.padding_side = "left"  # Allow batched inference

        self.max_length = max_length
        self.short_template = short_template
        self.long_template = long_template
        
    def forward(self, xs, template_type='short', multi_template=False, template=None, template_num=7, infer=True):
        decoded = []
        for i in range(xs.shape[0]):
            x = xs[i]
            x = x[x != self.tokenizer.pad_token_id]
            decoded.append(x)
        xs = decoded
        xs = self.tokenizer.batch_decode(xs, skip_special_tokens=True)
        if template is None:
            template = self.long_template
            if not multi_template:
                template = self.short_template
        template = template.replace('_', ' ').replace('*sep+*', '').replace('*cls*', '')

        for i, s in enumerate(xs):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
            xs[i] = template.replace('*sent 0*', s).strip()
      
        if template_type == 'short':
            max_length = 1536
        else:
            max_length = 1536

        batch = self.tokenizer.batch_encode_plus(
            xs,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        ).to(self.model.device)

        input_ids = batch.input_ids
        embed_tokens = self.model.embed_tokens(input_ids)
        dtype = embed_tokens.dtype
        attention_mask = batch.attention_mask.clone()
        batch_size, seq_length = input_ids.shape

        if multi_template:
            print('embedding with multiple templates')
            embeddings = []

            attention_mask_ = torch.tril(torch.ones((batch_size, seq_length, seq_length), dtype=dtype, device=input_ids.device))
            attention_mask = attention_mask.unsqueeze(-1) * attention_mask_
            quote_token_id = input_ids[0, -1]
            
            quote_indices = torch.where(input_ids[0] == quote_token_id)[0]
            sep_sent = ['After thinking step by step , this image description means in just one word:"']
            sep_tok = self.tokenizer(
                sep_sent,
                return_tensors='pt'
            ).to(self.model.device)
            sep_token_id = sep_tok.input_ids[0, -10]
            sep_indices = torch.where(input_ids[0] == sep_token_id)[0]

            sentence_quote_indices = quote_indices
            print(sentence_quote_indices)

            last_n_quotes = sentence_quote_indices[-template_num:]
            first_sep_index = sep_indices[-1]

            print(last_n_quotes)
            print(first_sep_index)
            actual_template_num = len(last_n_quotes)

            assert actual_template_num == template_num, f"Warning: Only found {actual_template_num} quotes, but expected {template_num}"

            for i in range(template_num):
                if i == 0:
                    attention_mask[:, first_sep_index+1:last_n_quotes[i]+1, last_n_quotes[i]+1:] = 0
                else:
                    attention_mask[:, last_n_quotes[i-1]+1:last_n_quotes[i]+1, last_n_quotes[i]+1:] = 0
                    attention_mask[:, last_n_quotes[i-1]+1:last_n_quotes[i]+1, first_sep_index+1:last_n_quotes[i-1]+1] = 0

            num_heads = self.model.config.num_attention_heads
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_heads, seq_length, seq_length)

            attention_mask = attention_mask.masked_fill(attention_mask == 0, torch.finfo(dtype).min)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0)

            last_hidden_states = self.forward_hook(input_ids=input_ids, attention_mask=attention_mask)

            for i in range(template_num):
                embedding_last_quote = last_hidden_states[:, last_n_quotes[i], :]
                embeddings.append(embedding_last_quote.unsqueeze(1))

            embeddings = torch.cat(embeddings, dim=1)
            outputs = embeddings
            if infer:
                outputs = torch.mean(outputs, dim=1)

        else:
            print('embedding with single template')
            outputs = self.forward_hook(input_ids=input_ids, attention_mask=attention_mask)[:, -1, :]

        return outputs

    def forward_hook(self, input_ids, attention_mask):
        layers = self.embedding_layers.split('_')
        outputs_layers = {}

        def collect_layer_outputs(name):
            def hook(model, input, output):
                outputs_layers[name] = output
            return hook

        for i in layers:
            layer_name = f"model.layers.{int(i)-1}"
            try:
                self.get_submodule(layer_name).register_forward_hook(collect_layer_outputs(layer_name))
            except AttributeError:
                print(f"Layer {i} not found or naming convention does not match.")

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, :, :]
        
        outputs_mean = []
        for layer_name in outputs_layers:
            outputs_mean.append(outputs_layers[f'{layer_name}'][0][:, :, :].unsqueeze(0))
        for name, hook in self._forward_hooks.items():
            hook.remove()
        outputs_mean = torch.mean(torch.cat(outputs_mean, dim=0), dim=0)

        return outputs_mean

    @staticmethod
    def from_pretrained(base_model_path, layer, tokenizer, max_length, short_template, long_template, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True, 
            **kwargs
        )

        return MultiPromptEOLEmbeddingsLLM(base_model, layer, tokenizer, max_length, short_template, long_template)