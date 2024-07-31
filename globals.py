from ipywidgets import interact
import ipywidgets as widgets
import torch

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
                              learning_rate = widgets.FloatLogSlider(value = 6e-4, base = 10, min = -5, max = -2, step = 0.001, description = "Learning Rate", style=style, layout = widgets.Layout(width='50%')),
                              weight_decay = widgets.FloatLogSlider(value = 1e-1, base = 10, min = -5, max = -1, step = 0.01, description = "Weight Decay", style=style, layout = widgets.Layout(width='50%')),
                              batch_size = widgets.IntSlider(value = 16, min = 1, max = 64, description = "Batch Size", style=style, layout = widgets.Layout(width='50%')),
                              grad_accum_steps = widgets.IntSlider(value = 30, min = 1, max = 128, description = "Gradient Accumulation Steps", style=style, layout = widgets.Layout(width='50%')),
                              max_steps = widgets.IntSlider(value = 10000, min = 2500, max = 600000, description = "Training Steps", style=style, layout = widgets.Layout(width='50%')),
                              checkpoint = widgets.IntSlider(value = 1000, min = 1, max = 10000, description = "Checkpoint Freq", style=style, layout = widgets.Layout(width='50%')),
                              starting_point = widgets.Text(value = "", placeholder=  "ckpt9999.pth", description = "Load from checkpoint: ", style=style, layout = widgets.Layout(width='50%')),
                              )
        
        
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

