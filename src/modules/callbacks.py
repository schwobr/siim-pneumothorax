from itertools import tee
import torch
from fastai.basic_data import DeviceDataLoader
# from fastai.basic_train import Learner
from fastai.callback import OptimWrapper
from fastai.callbacks import LearnerCallback
from dataclasses import dataclass
import neptune


def new_iter(self):
    """
    Replaces DeviceDataLoader.__iter__ so that the iterator over sampler
    indexes is doubled and can be accessed in UpdateSamplerCallback.
    """
    dl = iter(self.dl)
    dl.sampler_iter, self.sampler_iter = tee(dl.sampler_iter)
    for b in dl:
        yield self.proc_batch(b)


DeviceDataLoader.__iter__ = new_iter


class UpdateSamplerCallback(LearnerCallback):
    """
    Callback that works with modules.samplers.RandomSampler.
    Updates sampler's weights depending on loss for each item
    """
    _order = 0

    def on_backward_begin(self, **kwargs):
        """
        Gets the unreduced loss stored in modules.losses.URLoss,
        gets the indexes of the last batch and updates the weights
        of the sampler with the corresponding losses.
        """
        loss = self.learn.loss_func.loss.float().detach()
        if len(loss.shape) > 1:
            loss = loss.view(loss.size(0), -1)
            loss = loss.mean(-1)
        dl = self.learn.data.train_dl
        idxs = next(dl.sampler_iter)
        dl.sampler.weights[idxs] = 1-torch.exp(-loss.cpu())
        dl.sampler.to_update[idxs] = False

    def on_epoch_end(self, **kwargs):
        """
        The weights of the items that weren't used are doubled.
        """
        sampler = self.learn.data.train_dl.sampler
        sampler.weights[sampler.to_update] *= 2
        sampler.weights = torch.clamp(sampler.weights, 0., 1.)
        sampler.to_update = torch.ones_like(sampler.weights, dtype=torch.bool)


class AccumulateOptimWrapper(OptimWrapper):
    def step(self): pass
    def zero_grad(self): pass
    def real_step(self):      super().step()
    def real_zero_grad(self): super().zero_grad()


def acc_create_opt(self, lr, wd=0.):
    "Create optimizer with `lr` learning rate and `wd` weight decay."
    self.opt = AccumulateOptimWrapper.create(
        self.opt_func, lr, self.layer_groups, wd=wd,
        true_wd=self.true_wd, bn_wd=self.bn_wd)


# Learner.create_opt = acc_create_opt


@dataclass
class AccumulateStep(LearnerCallback):
    """
    Does accumlated step every nth step by accumulating gradients
    """

    def __init__(self, learn, n_step=1):
        super().__init__(learn)
        self.n_step = n_step

    def on_epoch_begin(self, **kwargs):
        "init samples and batches, change optimizer"
        self.acc_batches = 0

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "accumulate samples and batches"
        self.acc_batches += 1

    def on_backward_end(self, **kwargs):
        "step if number of desired batches accumulated, reset samples"
        if (self.acc_batches % self.n_step) == self.n_step - 1:
            for p in (self.learn.model.parameters()):
                if p.requires_grad:
                    p.grad.div_(self.acc_batches)

            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0

    def on_epoch_end(self, **kwargs):
        "step the rest of the accumulated grads"
        if self.acc_batches > 0:
            for p in (self.learn.model.parameters()):
                if p.requires_grad:
                    p.grad.div_(self.acc_batches)
            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0


class MonitorGrad(LearnerCallback):
    """
    Callback used to monitor gradients
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grads = []

    def on_backward_end(self, **kwargs):
        """
        Gets all gradients and computes the mean
        """
        grads = []
        for p in self.learn.model.parameters():
            if p.grad is not None:
                grad = p.grad.float().mean()
                grads.append(grad.item())
        mean_grad = sum(grads)/len(grads)
        self.grads.append(mean_grad)


class MTLLossCallback(LearnerCallback):
    """
    Callback used for multi-task learning weight update
    See arXiv:1705.07115
    """
    _order = -100

    def on_loss_end(self, **kwargs):
        """
        Gets the different losses stored in modules.losses.MTLLoss
        and weight them using log_vars stoed in modules.models.MultiTaskModel.
        Returns the mean of the losses.
        """
        losses = self.learn.loss_func.losses
        losses = [
            loss * torch.exp(-log_var) + log_var / 2 for log_var,
            loss in zip(self.learn.model.log_vars.val, losses)]
        return {'last_loss': sum(losses)/len(losses)}


class NeptuneCallback(LearnerCallback):
    """
    Callback used to send data to neptune.ml

    learner: Learner object to monitor
    project: neptune Project to create the experiment on
    name: name of the experiment
    params: hyperparameters to send
    kwargs: optional LearnerCallback kwargs
    """
    _order = 500

    def __init__(
            self, learner, project, name='Untitled', params={},
            **kwargs):
        super().__init__(learner, **kwargs)
        neptune.init('schwobr/SIIM-Pneumothorax')
        self.exp = project.create_experiment(name=name, params=params)

    def on_backward_begin(self, last_loss, iteration, **kwargs):
        self.exp.send_metric('train_loss', iteration, last_loss)

    def on_epoch_end(self, last_metrics, epoch, **kwargs):
        metric_names = [met.__name__ for met in self.learn.metrics]
        self.exp.send_metric('valid_loss', epoch, last_metrics[0])
        for m, v in zip(metric_names, last_metrics[1:]):
            self.exp.send_metric(m, epoch, v)

    def stop(self):
        self.exp.stop()

    def send_artifact(self, *args, **kwargs):
        self.exp.send_artifact(*args, **kwargs)


class LookAhead(LearnerCallback):
    def __init__(self, *args, k=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def on_step_end(self, iteration, **kwargs):
        if iteration == 0:
            return
        if iteration % self.k == 0:
            self.learn.opt.look_ahead()
