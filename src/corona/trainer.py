import torch
from torch import save
from torch.distributed import init_process_group, get_rank, get_world_size
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler

import logging


class Trainer:
    loss_fn = MSELoss()

    def __init__(self, model, training_set, batch_size, learning_rate, validation_set=None, checkpoint_dir=None, sampler=None):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.sampler=sampler
        self.data_loader = self._get_data_loader(training_set, batch_size=batch_size, sampler=sampler)
        self.validation_data_loader = self._get_data_loader(validation_set, batch_size=1, shuffle=False)
        self.optimizer = self._get_optimizer(model, learning_rate)

    @staticmethod
    def _get_optimizer(model, lr):
        return Adam(model.parameters(), lr=lr)

    @staticmethod
    def _get_data_loader(dataset, batch_size=1, shuffle=True, sampler=None):
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None and shuffle), sampler=sampler)

    def train(self, epochs):
        training_history = []
        validation_history = []
        for epoch in range(epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            logging.info(f"{self.rank} Start Epoch {epoch}")
            loss_avg = self._train_step()
            training_history.append(loss_avg)
            self._print_loss(epoch, loss_avg)

            if self.validation_data_loader is not None:
                validation_loss = self._validation_step()
                validation_history.append(validation_loss)
                self._print_loss(epoch, validation_loss, type="validation")

            if epoch % 10 == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch, loss_avg)

        return self.model.eval(), training_history, validation_history

    def _train_step(self):
        self.model.train()
        loss_sum = 0
        for idx, (train_data, train_labels) in enumerate(self.data_loader):
            logging.debug(f"{self.rank} Batch Nr: {idx}")
            train_data = train_data.requires_grad_()
            self.optimizer.zero_grad()
            y_pred = self.model(train_data)
            loss = self.loss_fn(y_pred.float(), train_labels)
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()
        loss_avg = loss_sum / (len(self.data_loader) * self.data_loader.batch_size)
        return loss_avg

    def _validation_step(self):
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            for valid_data, valid_label in self.validation_data_loader:
                output = self.model(valid_data)
                validation_loss += self.loss_fn(output, valid_label).item()
                logging.debug(f"Model predicted {output.item()}; Correct was {valid_label.item()}")

        validation_loss /= len(self.validation_data_loader.dataset)
        return validation_loss

    def _reset_hidden_state(self):
        self.model.reset_hidden_state()

    def _print_loss(self, epoch, loss, type="train"):
        logging.info(f'Epoch {epoch} {type} loss: {loss}')

    def _save_checkpoint(self, epoch, loss):
        if self.checkpoint_dir is None:
            return
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir()

        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss
        }
        save(checkpoint, self.checkpoint_dir / f'checkpoint-epoch-{epoch}.pt')


class DDPTrainer(Trainer):
    def __init__(self, model, training_set, batch_size, learning_rate, validation_set=None, checkpoint_dir=None):
        init_process_group('mpi')
        self.rank = get_rank()
        self.world_size = get_world_size()
        super().__init__(
            model=DistributedDataParallel(model),
            training_set=training_set,
            validation_set=self._get_validation_set(validation_set),
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            sampler=DistributedSampler(training_set, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        )

    def _get_validation_set(self, validation_set):
        if self.rank is None or self.rank == 0:
            return validation_set
        else:
            return None

    def _reset_hidden_state(self):
        self.model.module.reset_hidden_state()

    def _print_loss(self, epoch, loss, type="train"):
        logging.info(f'{self.rank}: Epoch {epoch} {type} loss: {loss}')

    def _save_checkpoint(self, epoch, loss):
        if self.rank != 0:
            return
