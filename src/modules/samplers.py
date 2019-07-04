import torch
from torch.utils.data import Sampler


class RandomSampler(Sampler):
    def __init__(self, model, loss, dl, num_samples, bs):
        assert loss.reduction == 'none', 'Loss function must disable reduction'
        self.model = model
        self.bs = bs
        self.dl = dl.new(shuffle=False, sampler=None)
        self.loss = loss
        self.num_samples = num_samples

    def get_scores(self):
        losses = []
        with torch.no_grad():
            for X, y_true in self.dl:
                y_pred = self.model(X)
                loss = self.loss(y_pred, y_true)
                losses.append(loss)
        return torch.cat(losses)

    def __iter__(self):
        scores = self.get_scores()
        return iter(
            torch.multinomial(scores, self.num_samples, True).tolist())

    def __len__(self):
        return self.num_samples
