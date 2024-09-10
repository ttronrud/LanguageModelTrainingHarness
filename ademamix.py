"""
AdEMAMix optimizer from https://arxiv.org/abs/2409.03137

GH repo: https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch

Utilizes an additional EMA term to retain older gradient information
"""

import math
import torch
from torch.optim import Optimizer

class AdEMAMix(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), eps=1e-8,
                 weight_decay=0, alpha=5.0, T_alpha_beta3=None, foreach = True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        assert len(betas) == 3, f"Invalid beta parameters: {betas}, expected 3"
        assert all(0.0 <= beta < 1.0 for beta in betas), f"Invalid beta parameters: {betas}"
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        self.foreach = foreach
        if foreach:
            self.optim_fn = self._multi_tensor_ademamix
        else:
            self.optim_fn = self._update_adamemix
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha=alpha, T_alpha_beta3=T_alpha_beta3)
        super(AdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_slow = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdEMAMix does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.0, dtype=p.grad.dtype, device=p.device) if self.foreach else 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Slow exponential moving average
                        state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    exp_avg_slow.append(state['exp_avg_slow'])
                    state['step'] += 1
                    state_steps.append(state['step'])

            beta1, beta2, beta3 = group['betas']
            alpha = group['alpha']
            T_alpha_beta3 = group['T_alpha_beta3']
            
            self.optim_fn(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avg_slow,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                alpha=alpha,
                T_alpha_beta3=T_alpha_beta3,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
            )
        return loss

    def _update_adamemix(self, params, grads, exp_avgs, exp_avg_sqs, exp_avg_slow, state_steps,
                         beta1, beta2, beta3, alpha, T_alpha_beta3, lr, weight_decay, eps):
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_slow_i = exp_avg_slow[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if T_alpha_beta3 is not None:
                alpha_t = min(step * alpha / T_alpha_beta3, alpha)
                beta3_t = min(math.exp(math.log(beta1) * math.log(beta3) / 
                              ((1 - step / T_alpha_beta3) * math.log(beta3) + 
                               (step / T_alpha_beta3) * math.log(beta1))), beta3)
            else:
                alpha_t = alpha
                beta3_t = beta3

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            exp_avg_slow_i.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            if weight_decay != 0:
                param.add_(param, alpha=-weight_decay * lr)

            param.addcdiv_(exp_avg + alpha_t * exp_avg_slow_i, denom, value=-step_size)
            
            
    """
    multi-tensor AdEMAMix implementation, in the vein of
    torch.optim's _multi_tensor_adamw
    
    I've noticed some slight performance gains, but nothing that
    absolutely blows the original out of the water. 
    Here's to waiting for fused kernels...
    """    
    def _multi_tensor_ademamix(self, params, grads, exp_avgs, exp_avg_sqs, exp_avg_slow, state_steps,
                         beta1, beta2, beta3, alpha, T_alpha_beta3, lr, weight_decay, eps):
        if len(params) == 0:
            return
        
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
            [params, grads, exp_avgs, exp_avg_sqs, exp_avg_slow, state_steps]
        )
        for (
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            exp_avg_slow,
            device_state_steps,
        ), _ in grouped_tensors.values():
            
            if T_alpha_beta3 is not None:
                alpha_t = min(step * alpha / T_alpha_beta3, alpha)
                beta3_t = min(math.exp(math.log(beta1) * math.log(beta3) / 
                              ((1 - step / T_alpha_beta3) * math.log(beta3) + 
                               (step / T_alpha_beta3) * math.log(beta1))), beta3)
            else:
                alpha_t = alpha
                beta3_t = beta3
            
            torch._foreach_add_(device_state_steps, 1)
            
            if weight_decay != 0:
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            
            # Decay the first and second moment running average coefficient
            torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
            torch._foreach_mul_(device_exp_avg_sqs, beta2)
            torch._foreach_lerp_(exp_avg_slow, device_grads, 1 - beta3_t)
            
            torch._foreach_addcmul_(
                device_exp_avg_sqs, device_grads, device_grads, 1 - beta2
            )
            
            del device_grads
                        
            bias_correction1 = [
                1 - beta1 ** step.item() for step in device_state_steps
            ]
            bias_correction2 = [
                1 - beta2 ** step.item() for step in device_state_steps
            ]
            
            
            
            if not torch.jit.is_scripting() and torch._utils.is_compiling():
                step_size = torch.stack([(lr / bc) * -1 for bc in bias_correction1])
            else:
                step_size = [(lr / bc) * -1 for bc in bias_correction1]

            bias_correction2_sqrt = [
                bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
            ]
            
            exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
            
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            
            torch._foreach_add_(device_exp_avgs, exp_avg_slow, alpha=alpha_t)
            torch._foreach_addcdiv_(
                device_params,
                device_exp_avgs,
                exp_avg_sq_sqrt,
                step_size,
            )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    