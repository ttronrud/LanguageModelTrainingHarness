import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import tiktoken
from ademamix import AdEMAMix

def clean_class_name(st):
    return st.replace("__main__.","")

"""
Rotary Embedding utils from Transformers LLama source
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L152
"""
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    

"""
KV interleave utils from Transformers LLama source
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L220
"""
def repeat_kv(hidden_states, n_rep):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Transformer(nn.Module):
    """
    Utility functions from Karpathy's nanoGPT
    https://github.com/karpathy/nanoGPT/blob/master/model.py#L118
    
        Isolated here to keep notebook code cleaner and more focused
        on the architecture
    """

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    """
    Karpathy's weight initialization scheme -- seems pretty much standard
    across GPT, LLama, etc.
    """
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    """
    Karpathy's code to configure an optimizer (e.g. AdamW) for a model, to handle 
    weights and biases/layernorms differently.
    """
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_adamw = False):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        if use_adamw:
            # AdamW optimizer - has conveniences like fused kernels and integrated support
            
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and 'cuda' in device_type
            extra_args = dict(fused=True) if use_fused else dict()
            
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using AdamW - Fused: {use_fused}")
        else:
            # AdEMAMix optimizer - utilizes a third beta term to retain
            # old gradients while also incorporating new gradient information
            # to supposedly ~double training effectiveness
            optimizer = AdEMAMix(optim_groups, lr=learning_rate, betas=betas)
            print(f"using AdEMAMix")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None, min_p = 0, stop_tokens = []):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # optionally crop the least-probable logits based on the 
            # scale of the most probable. "min-P" sampling strategy
            if min_p > 0:
                pmax, maxind = torch.max(probs, dim = -1)
                pmin = min_p * pmax
                for i,ps in enumerate(probs):
                    probs[i,ps < pmin] = 0
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next in stop_tokens:
                break

        return idx
        
    
"""
Text colour methods 
"""
def prRed(s): return "\033[91m {}\033[00m" .format(s)
def prGreen(s): return "\033[92m {}\033[00m" .format(s)
def prYellow(s): return "\033[93m {}\033[00m" .format(s)
def prLightPurple(s): return "\033[94m {}\033[00m" .format(s)
def prPurple(s): return "\033[95m {}\033[00m" .format(s)
def prCyan(s): return "\033[96m {}\033[00m" .format(s)
def prLightGray(s): return "\033[97m {}\033[00m" .format(s)
def prBlack(s): return "\033[98m {}\033[00m" .format(s)

def pretty_generate(instring, model, top_k = 50, min_p = 0, temperature = 1.0, gen_len = 200, 
                    seed = None, enc = "gpt2",genColFunc = prGreen, dev = "cpu"):
    enc = tiktoken.get_encoding(enc)
    
    #Model expects shape of (batch_size, seq_len) so we need to use a view with batch_size of 1
    intok = torch.Tensor(enc.encode_ordinary(instring)).view(1,-1).long().to(dev)
    #Allow setting of specific RNG seed for reproduceability
    if seed != None:
        torch.manual_seed(seed)
    else:
        seed = torch.randint(low=0,high=999999,size=(1,)).item()
        torch.manual_seed(seed)
        print(f"Generating with random seed {seed}")
    
    #Use the generate method to start writing tokens
    out_tok = model.generate(intok, gen_len, top_k = top_k, temperature = temperature, min_p = min_p)
    
    #move output off GPU, and slice off only newly generated text
    #tiktoken expects "flat" list, so we view it like that
    out_str = enc.decode(out_tok.to("cpu").view(-1).tolist())[len(instring):]
    return prBlack(instring) + genColFunc(out_str)