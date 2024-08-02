# Language Model Training Harness
#### Thorold Tronrud, ML/AI Software Engineer, StarFish Medical

## Introduction
The aim of this project is to distill a lot of modern language modelling into a principally educational and/or demonstrative jupyter notebook. Regardless of ones opinions on notebooks themselves, the unification of both markdown and code cells is the most accessible approach to take to display information alongside fully-functional code. To this end, modelling and training utilities that I considered less architecturally-relevant are placed inside utility Python files, while what I considered the core of the system -- namely the Transformer stack, and training loop live in the main notebook.

I opted to make the dependencies as low-level as feasible (e.g. at the Torch level) to better demonstrate the individual aspects of the architecture. However, many of the code blocks were adapted directly *from* [Transformers](https://github.com/huggingface/transformers), albeit with modifications (and citations), under the "if it ain't broke" rule of thumb. In a similar vein, I adpated pieces of [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) project, which is another excellent end-to-end example, though narrowly focused on the GPT-2 architecture. 

The primary goal was to investigate "one GPU in one day" (with the caveat that the "GPU" in question is a 4090), though all manner of training setup are possible, if not necessarily recommended...

## Dependencies
The notebook was developed in Python 3.11, with PyTorch 2.3.1, however there are unlikely to be breaking changes introduced in newer versions of these libraries, and it's even possible that performance improvements are introduced (particularly with new versions of FlashAttention). The minimal required environment can be installed with:
```
pip install torch numpy matplotlib datasets tiktoken wandb tqdm torchinfo notebook ipywidgets
```
with the latter two dependencies required for the notebook itself.

## Usage
The **LMHarness** notebook should be your primary entry-point, and is a full end-to-end foundational model training system. By default, execution through the notebook will result in a ~220M parameter LLama-3 style model, replete with RoPE, GQA, and gated FFNs. Through configuration, it's possible to not only vary the size, both "vertically" and "horizontally", but also the attention mechanism (raw MHA, RoPE MHA, RoPE GQA), positional embedding style (Absolute, Rotary, and NoPE), and Feed Forward style (both MLP and Gated MLP). 

The code is oriented towards plug-and-play, to allow experimentation with different setups -- and includes optional Weights & Biases reporting. 

## Experimentation
While the feasible experimentation with my hardware setup is... Limited, especially in terms of model scale and training length (we can't all have 8xH100s on hand), I have been able to do some basic setup comparisons on the low end of both. My findings can be briefly summarized as:

- After the final "Phase Change" ([See Here](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#where-are-induction-heads-located-in-models)) model training loss converges based on total parameter size. Width vs verticality (embedding and intermediate size vs number of layers) has the greatest effect only very early in training, and *potentially* later on.
- Size matters, but not necessarily as much as you think, especially at this scale. For similar total training times (on the order of 1 day to 1 week), a larger ~1B parameter model will likely reach a similar training loss as a smaller model, which will have "seen" more data. This has been borne out by Meta, with their emphasis on "15T training tokens" and corresponding excellent performance, even for "smaller" 8B paramter models. Model size does matter if you're constraining training by total number of tokens in your available training data.
