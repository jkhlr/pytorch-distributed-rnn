import logging

import torch
from torch import save
from torch.distributed import init_process_group, get_rank, get_world_size
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler


class Trainer:
    loss_fn = CrossEntropyLoss()

    def __init__(self, model, training_set, batch_size, learning_rate, validation_set=None, test_set=None,
                 checkpoint_dir=None, sampler=None):
        if not hasattr(self, 'rank'):
            self.rank = 0
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.sampler = sampler
        self.formatter = None  # set in train
        self.train_loader = self._get_data_loader(training_set, batch_size=batch_size, sampler=sampler)
        self.validation_loader = self._get_data_loader(validation_set, batch_size=batch_size, shuffle=False)
        self.test_loader = self._get_data_loader(test_set, batch_size=batch_size, shuffle=False)
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
        self.formatter = TrainingMessageFormatter(self.rank, epochs)
        training_history = []
        validation_history = []
        best_loss = None

        for epoch in range(epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            logging.info(f"{self.rank} Start Epoch {epoch + 1}")
            train_loss, train_acc = self._train_step()
            training_history.append(train_loss)

            if self.validation_loader is not None:
                validation_loss, val_acc = self._evaluate(self.validation_loader, epoch)
                validation_history.append(validation_loss)

                if best_loss is None or best_loss > validation_loss:
                    best_loss = validation_loss
                    self._save_checkpoint(epoch, validation_loss)

            if self.rank == 0 and (epoch % 10 == 0 or epoch == epochs - 1):
                self._save_checkpoint(epoch, train_loss)

        if self.test_loader is not None:
            self._evaluate(self.validation_loader)

        return self.model.eval(), training_history, validation_history

    def _train_step(self):
        self.model.train()
        total_loss = 0.
        total_correct = 0
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.requires_grad_()
            labels = labels.long().squeeze(1)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, labels)
            total_loss += loss.item()
            correct = (torch.argmax(output, dim=1) == labels).sum()
            total_correct += correct
            loss.backward()
            self.optimizer.step()
            logging.info(self.formatter.train_progress_message(batch_idx=batch_idx, batches=len(self.train_loader),
                                                               training_examples=len(data), correct=correct,
                                                               loss=loss.item()))

        train_loss = total_loss / len(self.train_loader.dataset)
        train_acc = total_correct / len(self.train_loader.dataset)
        return train_loss, train_acc

    def _evaluate(self, data_loader, epoch=None):
        self.model.eval()
        eval_loss = 0.
        total_correct = 0
        with torch.no_grad():
            for data, labels in data_loader:
                output = self.model(data)
                labels = labels.long().squeeze(1)
                eval_loss += self.loss_fn(output, labels).item()
                total_correct += (torch.argmax(output, dim=1) == labels).sum()
                logging.debug(f"Model predicted {torch.argmax(output, dim=1).data}; Correct was {labels.data}")

        eval_loss /= len(data_loader)
        num_examples = len(data_loader.dataset)
        accuracy = float(total_correct) / num_examples

        logging.info(self.formatter.evaluation_message(accuracy, num_examples, epoch, eval_loss, total_correct))
        return eval_loss, accuracy

    def _reset_hidden_state(self):
        self.model.reset_hidden_state()

    def _save_checkpoint(self, epoch, loss, best=False):
        epoch += 1
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
        name = "best-model.pt" if best else f'checkpoint-epoch-{epoch}.pt'
        save(checkpoint, self.checkpoint_dir / name)


class DDPTrainer(Trainer):
    def __init__(self, model, training_set, batch_size, learning_rate, validation_set=None, test_set=None,
                 checkpoint_dir=None):
        init_process_group('mpi')
        self.rank = get_rank()
        self.world_size = get_world_size()
        super().__init__(
            model=DistributedDataParallel(model),
            training_set=training_set,
            validation_set=self._get_eval_set(validation_set),
            test_set=self._get_eval_set(test_set),
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            sampler=DistributedSampler(training_set, num_replicas=self.world_size, rank=self.rank, shuffle=True),
        )

    def _get_eval_set(self, eval_set):
        if self.rank is None or self.rank == 0:
            return eval_set
        else:
            return None

    def _reset_hidden_state(self):
        self.model.module.reset_hidden_state()

    def _save_checkpoint(self, epoch, loss, best=False):
        if self.rank != 0:
            return


class TrainingMessageFormatter:
    def __init__(self, rank, num_epochs):
        self.rank = rank
        self.num_epochs = num_epochs

    def train_progress_message(self, batch_idx, batches, training_examples, correct, loss):
        batch_idx += 1
        return 'Rank: {:02d}   Train Batch: {}/{} ({:.0f}%)\tLoss: {:.6f}\tAcc: {}/{} ({:.0f}%)' \
            .format(self.rank, batch_idx, batches, percentage(batch_idx, batches), loss, correct, training_examples,
                    percentage(float(correct), training_examples))

    def evaluation_message(self, accuracy, examples, epoch, eval_loss, total_correct):
        metrics = 'Loss: {:.4f}\t Accuracy: {}/{} ({:.0f}%)\n' \
            .format(eval_loss, total_correct, examples, 100. * accuracy)
        if epoch is None:
            prefix = "Test Evaluation:\t"
        else:
            epoch += 1
            prefix = 'Evaluation Epoch: {}/{} ({:.0f}%)\t' \
                .format(epoch, self.num_epochs, percentage(epoch, self.num_epochs))
        return prefix + metrics


def percentage(current, overall):
    return 100. * (current / overall)
