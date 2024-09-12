from ipywidgets import interact
import ipywidgets as widgets
import torch
import psutil

SHOW_GRADIENT = True
USE_WANDB = False

LR = 6e-4
WD = 1e-1
E = 1
MAX_STEPS = 10000
BATCH_S = 16
GRAD_ACCUM_STEPS = 30
CHKPT_FREQ = 1000
plotfreq = 10
loadchkpt = None
modelmegaparams = 220


dev = "cpu"
device = None
dtype = torch.float16

train_controls = None

style = {'description_width': 'initial'}
def parseTrainOptions(showgradient = True, 
                      usewandb = False, 
                      learning_rate = 6e-4, 
                      weight_decay = 1e-1,
                      batch_size = 16, 
                      grad_accum_steps = 30, 
                      max_steps = 10000,
                      checkpoint = 1000,
                      starting_point = ""):
    global SHOW_GRADIENT
    global USE_WANDB
    global LR
    global WD
    global MAX_STEPS
    global BATCH_S
    global GRAD_ACCUM_STEPS
    global CHKPT_FREQ
    global loadchkpt
    
    SHOW_GRADIENT = showgradient
    USE_WANDB = usewandb
    LR = learning_rate
    WD = weight_decay
    MAX_STEPS = max_steps
    BATCH_S = batch_size
    GRAD_ACCUM_STEPS = grad_accum_steps
    CHKPT_FREQ = checkpoint
    loadchkpt = starting_point

def train_control_setup():
    global train_controls
    train_controls = interact(parseTrainOptions, 
                              showgradient = widgets.Checkbox(value = True, description = "Show Gradients"),
                              usewandb = widgets.Checkbox(value = False, description = "Use W&B"),
                              learning_rate = widgets.FloatLogSlider(value = 6e-4, base = 10, min = -5, max = -2, step = 0.001, description = "Learning Rate", style=style, layout = widgets.Layout(width='75%')),
                              weight_decay = widgets.FloatLogSlider(value = 1e-1, base = 10, min = -5, max = -1, step = 0.01, description = "Weight Decay", style=style, layout = widgets.Layout(width='75%')),
                              batch_size = widgets.IntSlider(value = BATCH_S, min = 1, max = 4*BATCH_S, description = "Batch Size", style=style, layout = widgets.Layout(width='75%')),
                              grad_accum_steps = widgets.IntSlider(value = GRAD_ACCUM_STEPS, min = 1, max = 2*GRAD_ACCUM_STEPS, description = "Gradient Accumulation Steps", style=style, layout = widgets.Layout(width='75%')),
                              max_steps = widgets.IntSlider(value = 10000, min = 2000, max = 600000, step=1000, description = "Training Steps", style=style, layout = widgets.Layout(width='75%')),
                              checkpoint = widgets.IntSlider(value = 1000, min = 10, max = 10000, step=10, description = "Checkpoint Freq", style=style, layout = widgets.Layout(width='75%')),
                              starting_point = widgets.Text(value = "", placeholder=  "ckpt9999.pth", description = "Load from checkpoint: ", style=style, layout = widgets.Layout(width='75%')),
                              )
                              
"""
Guesstimate a batch size that won't exceed available VRAM
using torchinfo stats, and targeting 500k tokens per backprop
"""
def estimate_batch_size_globals(model_stats, target_tokens = 500000, seq_len = 1024, gradient_checkpointing = True):
    global dev
    global BATCH_S
    global GRAD_ACCUM_STEPS
    
    if "cuda" in dev:
        max_mem = torch.cuda.mem_get_info()[0]
    else:
        meminf = psutil.virtual_memory()
        max_mem = meminf.available
        print(f"Using CPU and system RAM is *not* recommended for training. {max_mem} bytes available.")
    # try to estimate VRAM usage, scale back batch size and scale update
    # accumulation steps to try to make it fit...
    total_params = model_stats.total_param_bytes
    input_bytes = model_stats.total_input
    output_bytes = model_stats.total_output_bytes
    total_bytes = total_params + input_bytes + output_bytes
    
    BATCH_S = int(max_mem/total_bytes)
    
    if gradient_checkpointing:
        BATCH_S *= 2
    
    BATCH_S = (BATCH_S//4 + 1)*4
        
    #next, how many batches do we need to run through to hit near target_tokens?
    GRAD_ACCUM_STEPS = target_tokens/(BATCH_S * seq_len)
        
def get_train_controls():
    global train_controls
    return train_controls.widget

def showGradient():
    global g_show_gradient
    return g_show_gradient
        
def useWandb():
    global g_use_wandb
    return g_use_wandb
    
def get_device():
    global dev
    global device
    global dtype
    
    if torch.cuda.is_available():
        dev = "cuda"
        device = torch.device(dev)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        dtype = torch.bfloat16
        print("Using cuda")
    else:  
        dev = "cpu"  
        device = torch.device(dev)
        
def set_device(d = "cuda:0"):
    global dev
    global device
    global dtype
    
    dev = d
    device = torch.device(dev)
    if "cuda" in d:
        dtype = torch.bfloat16
        print("Using cuda")

