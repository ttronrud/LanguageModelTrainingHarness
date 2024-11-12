import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import tiktoken

# Muon optimizer from
# https://github.com/KellerJordan/modded-nanogpt
# which I've found produces better results than
# straight AdamW
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5, weight_decay = 0,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # Perform stepweight decay
                p.data.mul_(1 - lr * weight_decay)
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < 200:
        return (it+1) / 200
    # 2) constant lr for a while
    return 1.0

class AdamW_Muon_Optimizer(torch.optim.Optimizer):
    def __init__(self, params, adamw_only = None, lr=3e-4, momentum=0.95, adamw_betas = (0.9, 0.95), muon_wd = 0, adamw_wd = 0,  nesterov=True, backend='newtonschulz5', backend_steps=5):
        
        
        muon_params = params
        adamw_params = adamw_only
        
        self.muon = Muon(muon_params, lr = 0.1*lr, momentum = momentum, nesterov = nesterov, backend = backend, backend_steps = backend_steps, weight_decay=muon_wd)
        
        self.adamw = torch.optim.AdamW(adamw_only, lr=lr, betas=adamw_betas,
                               weight_decay=adamw_wd, fused=True)
        
        self.optimizers = [self.muon, self.adamw]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in self.optimizers]
    
    def step(self):
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.step()
            sched.step()
    
    def get_state_dicts(self):
        sds = []
        for opt, sched in zip(self.optimizers, self.schedulers):
            sds.append({"optimizer":opt.state_dict(), "scheduler":sched.state_dict()})
        return sds
    def load_state_dicts(self, sds):
        for sdd, opt, sched in zip(sds, self.optimizers, self.schedulers):
            opt.load_state_dict(sdd["optimizer"])
            sched.load_state_dict(sdd["scheduler"])

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
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_adamw = True):
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
            
            optimizer = AdamW_Muon_Optimizer([{'params':decay_params}], adamw_only = [{'params':nodecay_params}], 
                                                lr = learning_rate, adamw_betas = betas, muon_wd = weight_decay)
            
            return optimizer
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