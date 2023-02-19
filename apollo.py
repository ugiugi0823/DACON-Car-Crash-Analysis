#https://github.com/XuezheMax/apollo/blob/master/optim/apollo.py
import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class Apollo(Optimizer):
    r"""Implements Atom algorithm.
        Arguments:
            params (iterable): 최적화할 매개변수의 반복 가능 또는 정의 정의
                parameter groups
            lr(float): 학습률
            beta (float, optional): 기울기의 실행 평균을 계산하는 데 사용되는 계수(기본값: 0.9)
            eps (float, optional): 수치적 안정성을 향상시키기 위해 분모에 추가된 용어(기본값: 1e-4)
            rebound(str, optional): 대각선 헤센에 대해 수정된 경계:
                ``'상수'`` | ``'belief'`` (기본값: 없음)
            warmup (int, optional): 워밍업 단계 수(기본값: 500)
            init_lr(float, optional): 워밍업을 위한 초기 학습률(기본값: lr/1000)
            weight_decay (float, optional): 가중치 감쇠 계수(기본값: 0)
            weight_decay_type (str, optional): 가중치 감쇠 유형:
                ``'L2'`` | ``'분리됨'`` | ``'안정적'`` (기본값: 'L2')
        """

    def __init__(self, params, lr, beta=0.9, eps=1e-4, rebound='constant', warmup=500, init_lr=None, weight_decay=0, weight_decay_type=None):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if rebound not in ['constant', 'belief']:
            raise ValueError("Invalid recitifed bound: {}".format(rebound))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type is None:
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError("Invalid weight decay type: {}".format(weight_decay_type))

        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super(Apollo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Apollo, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates(재평가) the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous update direction
                    state['update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Calculate current lr
                if state['step'] < group['warmup']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmup'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Atom does not support sparse gradients.')

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] == 'L2':
                    grad = grad.add(p, alpha=group['weight_decay'])

                beta = group['beta']
                eps = group['eps']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                state['step'] += 1
                bias_correction = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_correction

                # calc the diff grad
                delta_grad = grad - exp_avg_grad
                if group['rebound'] == 'belief':
                    rebound = delta_grad.norm(p=np.inf)
                else:
                    rebound = 0.01
                    eps = eps / rebound

                # Update the running average grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                denom = d_p.norm(p=4).add(eps)
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha) - B.mul(v_sq).sum()

                # Update B
                B.addcmul_(v_sq, delta)

                # calc direction of parameter updates
                if group['rebound'] == 'belief':
                    denom = torch.max(B.abs(), rebound).add_(eps / alpha)
                else:
                    denom = B.abs().clamp_(min=rebound)

                d_p.copy_(exp_avg_grad.div(denom))

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] != 'L2':
                    if group['weight_decay_type'] == 'stable':
                        weight_decay = group['weight_decay'] / denom.mean().item()
                    else:
                        weight_decay = group['weight_decay']
                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-curr_lr)

        return 
