import os

import torch
from ml_logger import logger

from .arrays import batch_to_device
from .timer import Timer


def cycle(dl):
    while True:
        for data in dl:
            yield data


class BCTrainer(object):
    def __init__(
        self,
        bc_model,
        dataset,
        train_batch_size=32,
        train_lr=2e-5,
        log_freq=100,
        save_freq=1000,
        eval_freq=10000,
        bucket=None,
        train_device="cuda",
        save_checkpoints=False,
    ):
        super().__init__()
        self.model = bc_model
        self.save_checkpoints = save_checkpoints

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq

        self.batch_size = train_batch_size

        self.dataset = dataset
        if dataset is not None:
            self.dataloader = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=train_batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )
            self.dataloader_vis = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )

        self.bucket = bucket
        self.optimizer = torch.optim.Adam(bc_model.parameters(), lr=train_lr)
        self.step = 0

        self.evaluator = None
        self.device = train_device

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def finish_training(self):
        if self.step % self.save_freq == 0:
            self.save()
        if self.eval_freq > 0 and self.step % self.eval_freq == 0:
            self.evaluate()
        if self.evaluator is not None:
            del self.evaluator

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        timer = Timer()
        for step in range(n_train_steps):
            batch = next(self.dataloader)
            batch = batch_to_device(batch, device=self.device)
            loss, infos = self.model.loss(*batch)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.save_freq == 0:
                self.save()

            if self.eval_freq > 0 and self.step % self.eval_freq == 0:
                self.evaluate()

            if self.step % self.log_freq == 0:
                logger.print(f"{self.step}: {loss:8.4f} | t: {timer():8.4f}")
                logger.log(step=self.step, loss=loss.detach().item(), flush=True)

            self.step += 1

    def evaluate(self):
        assert (
            self.evaluator is not None
        ), "Method `evaluate` can not be called when `self.evaluator` is None. Set evaluator with `self.set_evaluator` first."
        self.evaluator.evaluate(load_step=self.step)

    def save(self):
        """
        saves model and ema to disk;
        syncs to storage bucket if a bucket is specified
        """

        data = {
            "step": self.step,
            "model": self.model.state_dict(),
        }
        savepath = os.path.join(self.bucket, logger.prefix, "checkpoint")
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f"state_{self.step}.pt")
        else:
            savepath = os.path.join(savepath, "state.pt")
        torch.save(data, savepath)
        logger.print(f"[ utils/training ] Saved model to {savepath}")

    def load(self):
        """
        loads model and ema from disk
        """

        loadpath = os.path.join(self.bucket, logger.prefix, "checkpoint/state.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
