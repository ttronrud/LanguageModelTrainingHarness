import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
import tiktoken
import os
from tqdm import tqdm, trange
from datasets import Dataset, load_dataset, concatenate_datasets
import math
from torch.utils.data import DataLoader
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

enc = None
accesses = 0
memmap = None

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
def get_batch(fname = "train.bin", num = 1, block_size = 1024, batch_size = 16, dev = "cpu", accesses_per_refresh = 10):
    global memmap
    """
    Adapted to lower the number of accesses we make to the memory map before
    we re-generate it. As you access more parts of one memmap, RAM usage increases,
    so we're striking a balance between training speed/throughput and system memory
    usage.
    """
    if num % accesses_per_refresh == 0 or memmap is None:
        data_dir = ""
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        memmap = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    data = memmap
        
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
Creates a training data block - via numpy memmap, by streaming the target
dataset in. Parallelizes download/tokenization/writing and minimizes RAM
usage (or at least utilizes garbage collector effectively).

We re-open the memmap periodically because one instance will use more and more
RAM as it's accessed, up to the total block size. Refreshing the instance fixes this.

Started w/Karpathy's scheme, but made significant changes:
= uses streaming/iterable dataset interface to sidestep downloading
  the entire thing
= writes up to max tokens directly to memory mapped array as they come in
= generalizes better across many datasets/dataset configurations
""" 
def generate_training_data(dset = "HuggingFaceTB/smollm-corpus", dset_name = None, senc = "gpt2", num_proc = 1,
                            text_col = "text", out_fname = "train.bin", overwrite = False, tokens_to_save = 1e10):
    global enc
    
    enable_progress_bar()
    # leave if data binary already exists
    if os.path.isfile(out_fname) and overwrite == False:
        print(f"output file {out_fname} already exists. Set 'overwrite = True' to regenerate.")
        return
    if not os.path.exists("cache"):
        os.mkdir("cache")
    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(dset,dset_name, cache_dir = "cache", split = "train", streaming = True)
    dataset = iter(DataLoader(dataset, num_workers = 8, batch_size = 512, 
        collate_fn = lambda batch: default_collate([{"text":e.pop(text_col)} for e in batch])
        ))
    
    
    targtok = np.uint64(tokens_to_save)
    reopentok_inc = np.uint64(5e8)
    reopentok = reopentok_inc
    tot_tok = np.uint64(0)
    running = True
    
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(out_fname, dtype=dtype, mode='w+', shape=(int(targtok),))
    # have to disable progress bar here so terminal/cell isn't spammed
    # AND THERE'S NO 'verbose=False' OPTION
    disable_progress_bar()
    with tqdm(desc="Tokens processed", total = targtok, position=0, leave=True) as pbar:
        dsets = []
        while running:
            if tot_tok > reopentok:
                arr = np.memmap(out_fname, dtype=dtype, mode='r+')
                reopened = True
                reopentok += reopentok_inc

            dat = Dataset.from_dict(next(dataset))
            interim = dat.map(
                process,
                remove_columns=[text_col],
                desc="tokenizing the splits",
                num_proc=num_proc,
                #writer_batch_size = 100,
                #cache_file_names = {"train":"train.arrow"}
            )
            nu = np.sum(interim['len'], dtype=np.uint64)
            if tot_tok + nu < targtok:
                arr[tot_tok:tot_tok+nu] = np.concatenate(interim['ids'])
            else:
                arr[tot_tok:] = np.concatenate(interim['ids'])[:targtok - tot_tok]
            arr.flush()
            
            tot_tok += nu
            pbar.update(nu)
            if tot_tok > targtok:
                break
        enable_progress_bar()
    
    arr.flush()
    
"""
Operates the same as single-source training data generation
method, however it incorporates set counts of tokens from
multiple datasets, allowing for the construction of
composite training files.
"""     
def generate_training_data_mix(dset = ["HuggingFaceTB/smollm-corpus","togethercomputer/RedPajama-Data-1T"], 
                               dset_name = ["fineweb-edu-dedup", "arxiv"], 
                               text_col = ["text", "text"],
                               tokens_to_save = [1e10, 1e10],
                               senc = "gpt2", num_proc = 1,
                               out_fname = "train.bin", overwrite = False):
    global enc
    
    enable_progress_bar()
    # leave if data binary already exists
    if os.path.isfile(out_fname) and overwrite == False:
        print(f"output file {out_fname} already exists. Set 'overwrite = True' to regenerate.")
        return
    if not os.path.exists("cache"):
        os.mkdir("cache")
    enc = tiktoken.get_encoding("gpt2")
    
    tot_tok = np.uint64(0)
    reopentok_inc = np.uint64(5e8)
    reopentok = reopentok_inc
    end_tok = np.sum(tokens_to_save,dtype=np.uint64)
    
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(out_fname, dtype=dtype, mode='w+', shape=(int(end_tok),))
    
    for ds, ds_n, tc, ntok in zip(dset, dset_name, text_col, tokens_to_save):
        dataset = load_dataset(ds,ds_n, cache_dir = "cache", split = "train", streaming = True)
        dataset = iter(DataLoader(dataset, num_workers = 4, batch_size = 512, 
            collate_fn = lambda batch: default_collate([{"text":e.pop(tc)} for e in batch])
            ))
    
    
        tot_tok_ds = np.uint64(0)
        targtok = np.uint64(ntok)
        running = True
        
        # have to disable progress bar here so terminal/cell isn't spammed
        # AND THERE'S NO 'verbose=False' OPTION
        disable_progress_bar()
        with tqdm(desc="Tokens processed", total = targtok, position=0, leave=True) as pbar:
            dsets = []
            while running:
                if tot_tok > reopentok:
                    arr = np.memmap(out_fname, dtype=dtype, mode='r+')
                    reopened = True
                    reopentok += reopentok_inc

                dat = Dataset.from_dict(next(dataset))
                interim = dat.map(
                    process,
                    remove_columns=[tc],
                    desc="tokenizing the splits",
                    num_proc=num_proc
                )
                nu = np.sum(interim['len'], dtype=np.uint64)
                if tot_tok + nu >= end_tok:
                    arr[tot_tok:] = np.concatenate(interim['ids'])[:end_tok - tot_tok]
                    nu = end_tok - tot_tok
                elif tot_tok_ds + nu > targtok:
                    arr[tot_tok:tot_tok + (targtok - tot_tok_ds)] = np.concatenate(interim['ids'])[:targtok - tot_tok_ds]
                    nu = targtok - tot_tok_ds
                elif tot_tok + nu < end_tok:
                    arr[tot_tok:tot_tok+nu] = np.concatenate(interim['ids'])
                
                arr.flush()
                
                tot_tok += nu
                tot_tok_ds += nu
                pbar.update(nu)
                if tot_tok_ds > targtok:
                    break
            enable_progress_bar()
    
    arr.flush()
