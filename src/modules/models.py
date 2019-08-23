import torch.nn as nn
import torch
from fastai.vision.learner import unet_learner, cnn_learner
from fastai.callbacks.hooks import hook_outputs
from fastai.layers import flatten_model, ParameterModule
from fastai.torch_core import to_device


def set_BN_momentum(model, momentum=0.05):
    """
    Sets momentum of all batch norm layers of model

    model: model on which batch norm momentums are set
    momentum: momentum value
    """
    for layer in model.modules():
        if isinstance(
                layer, nn.BatchNorm2d) or isinstance(
                layer, nn.BatchNorm1d):
            layer.momentum = momentum


def forward(self, x):
    """
    Forward function for nn.ModuleList to allow to directly call
    subparts of fastai's Unet

    x: input tensor

    return: output tensor
    """
    res = x
    orig = x.orig
    for l in self:
        res.orig = orig
        nres = l(res)
        res.orig = None
        res = nres
    res.orig = orig
    return res


nn.ModuleList.forward = forward


class MultiTaskModel(nn.Module):
    """
    Base model for hard parameter sharing multi-task learning problem

    base: base model shared between tasks
    heads: task-specific models used on top of base
    """

    def __init__(self, base, heads, log_vars=None):
        super().__init__()
        self.base = base
        self.heads = nn.ModuleList(heads)
        if log_vars is None:
            log_vars = torch.zeros(len(self.heads))
        self.log_vars = ParameterModule(
            nn.Parameter(log_vars))

    def forward(self, x):
        """
        x: input tensor

        return: (N_tasks) list of output tensors for all tasks
        """
        res = x
        res.orig = x
        nres = self.base(res)
        res.orig = None
        res = nres
        all_res = []
        for head in self.heads:
            res.orig = x
            nres = head(res)
            all_res.append(nres)
            res.orig = None
        return all_res

    def __getitem__(self, i): return self.heads[i]
    def append(self, l): return self.heads.append(l)
    def extend(self, l): return self.heads.extend(l)
    def insert(self, i, l): return self.heads.insert(i, l)


def multi_task_unet_learner(*args, log_vars=None, **kwargs):
    """
    Creates a learner suited for classification+segmentation multii-task
    learning problem

    args: positional arguments for cnn_learner and unet_learner
    kwargs: keayword arguments for cnn_learner and unet_learner

    return: learner that contains MultiTaskModel
    """
    unet_learn = unet_learner(*args, **kwargs)
    sfs_idxs = unet_learn.model.sfs_idxs
    cnn_learn = cnn_learner(*args, **kwargs)
    base = unet_learn.model[0]
    unet_head = unet_learn.model[1:]
    hooks = hook_outputs([base[i] for i in sfs_idxs])
    for block, hook in zip(unet_head[3:7], hooks):
        block.hook = hook
    heads = [cnn_learn.model[1:], unet_head]
    unet_learn.model = MultiTaskModel(
        base, heads, log_vars=log_vars).to(
        unet_learn.data.device)
    lg = unet_learn.layer_groups
    lg[2] = nn.Sequential(
        *list(lg[2]),
        *flatten_model(heads[0]),
        unet_learn.model.log_vars)
    unet_learn.layer_groups = lg
    unet_learn.create_opt(slice(1e-3))
    return unet_learn


def replace_bn(learner, m, G=32):
    for n, ch in m.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            new_layer = to_device(
                nn.GroupNorm(G, ch.num_features, ch.eps, ch.affine),
                learner.data.device)
            m._modules[n] = new_layer
            found = False
            for lg in learner.layer_groups:
                for k, c in lg.named_children():
                    if c is ch:
                        lg._modules[k] = new_layer
                        found = True
                        break
                if found:
                    break
    for ch in m.children():
        replace_bn(learner, ch)

    if hasattr(learner, 'opt'):
        learner.create_opt(learner.opt.lr, wd=learner.wd)
