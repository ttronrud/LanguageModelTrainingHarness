import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from transformers import AutoTokenizer
import tiktoken
import os
from tqdm import tqdm, trange
from datasets import load_dataset
import math

enc = None

"""
Great visualization of gradients through model layers, found from:
https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
"""
def plot_grad_flow(named_parameters,f,ax):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    For residual networks like these, gradients are fairly robust throughout, so this
    just makes nice plots. Gratification >>> all
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow.
    Making a full copy of the gradients and shuffling them off to the CPU *is* a little
    intensive, so maybe don't do it every single step."""
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad != None:
            grad = p.grad.clone().cpu().float()
            layers.append(n)
            ave_grads.append(grad.abs().mean())
            max_grads.append(grad.abs().max())
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.set_xticks(range(0,len(ave_grads), 1), layers, rotation="vertical",size="xx-small")
    ax.set_xlim(left=0, right=len(ave_grads))
    #ax.set_ylim(top=0.02) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.set_yscale('log')
    ax.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])

"""
Karpathy's cosine LR scheduler w/warmup
"""
def get_lr(it, learning_rate=6e-4, min_lr=6e-5, warmup_iters=2000, lr_decay_iters=600000, decay_lr=True):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

"""
Karpathy's batch-fetching scheme
    
I don't think it's optimal, since having to load a memmap every batch fetch
is... Inelegant. There's definitely a better way to do this.
"""    
def get_batch(fname = "train.bin", block_size = 1024, batch_size = 16, dev = "cpu"):
    data_dir = ""
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
        
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    if dev == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x,y = x.pin_memory().to(torch.device(dev), non_blocking=True), y.pin_memory().to(torch.device(dev), non_blocking=True)
    else:
        x,y = x.to(torch.device(dev)), y.to(torch.device(dev))
    return x,y
    
    
"""
Karpathy's tokenizing mapped process
""" 
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token) 
    out =  {'ids': ids, 'len': len(ids)}
    return out
    
"""
Karpathy's dataset generation code, functionized.
""" 
def generate_training_data(dset = "openwebtext", senc = "gpt2", num_proc = 2,
                            text_col = "text", out_fname = "train.bin", overwrite = False):
    global enc
    
    # leave if data binary already exists
    if os.path.isfile(out_fname) and overwrite == False:
        print(f"output file {out_fname} already exists. Set 'overwrite = True' to regenerate.")
        return
    
    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(dset, num_proc=num_proc)
    # tokenize the dataset
    # added additional args to lower memory footprint
    tokenized = dataset.map(
        process,
        remove_columns=[text_col],
        desc="tokenizing the splits",
        num_proc=num_proc,
        writer_batch_size = 100,
        cache_file_names = {"train":"train.arrow"}
    )
    
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(out_fname, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {out_fname}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
