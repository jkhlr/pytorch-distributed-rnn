import logging
import time

import torch
from memory_profiler import memory_usage
from torch import save
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from trainer.formatter import TrainingMessageFormatter


class Trainer:
    loss_fn = CrossEntropyLoss()

    def __init__(self, model, training_set, batch_size, learning_rate,
                 validation_set=None, test_set=None,
                 checkpoint_dir=None, sampler=None):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.sampler = sampler
        self.train_loader = self._get_data_loader(training_set,
                                                  batch_size=batch_size,
                                                  sampler=sampler)
        self.validation_loader = self._get_data_loader(validation_set,
                                                       batch_size=batch_size,
                                                       shuffle=False)
        self.test_loader = self._get_data_loader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)
        self.optimizer = self._get_optimizer(model, learning_rate)

    def _get_optimizer(self, model, lr):
        return Adam(model.parameters(), lr=lr)

    def _get_data_loader(self, dataset, batch_size=1, shuffle=True,
                         sampler=None):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler
        )

    def _get_formatter(self, epochs):
        return TrainingMessageFormatter(epochs)

    def train(self, epochs):
        training_history = []
        validation_history = []
        formatter = self._get_formatter(epochs)

        def train_inner():
            best_loss = None

            for epoch in range(epochs):
                if self.sampler is not None:
                    self.sampler.set_epoch(epoch)
                logging.info(formatter.epoch_start_message(epoch))
                train_loss, train_acc = self._train_step(formatter)
                training_history.append(train_loss)

                if self.validation_loader is not None:
                    validation_loss, val_acc = self._evaluate(
                        self.validation_loader,
                        formatter,
                        epoch
                    )
                    validation_history.append(validation_loss)

                    if best_loss is None or best_loss > validation_loss:
                        logging.info(f"New best model in epoch {epoch + 1}")
                        best_loss = validation_loss
                        self._save_checkpoint(epoch, validation_loss, best=True)

                # if epoch % 10 == 0 or epoch == epochs - 1:
                #     self._save_checkpoint(epoch, train_loss)

        start = time.perf_counter()
        memory = max(memory_usage((train_inner, tuple(), {})))
        duration = time.perf_counter() - start
        logging.info(formatter.performance_message(memory, duration))

        if self.test_loader is not None:
            self._evaluate(self.test_loader, formatter)

        return self.model.eval(), training_history, validation_history

    def _train_step(self, formatter):
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
            logging.info(
                formatter.train_progress_message(
                    batch_idx=batch_idx,
                    batches=len(self.train_loader),
                    training_examples=len(data),
                    correct=correct,
                    loss=loss.item()
                )
            )

        train_loss = total_loss / len(self.train_loader.dataset)
        train_acc = total_correct / len(self.train_loader.dataset)
        return train_loss, train_acc

    def _evaluate(self, data_loader, formatter, epoch=None):
        self.model.eval()
        eval_loss = 0.
        total_correct = 0
        with torch.no_grad():
            for data, labels in data_loader:
                output = self.model(data)
                labels = labels.long().squeeze(1)
                eval_loss += self.loss_fn(output, labels).item()
                total_correct += (torch.argmax(output, dim=1) == labels).sum()
                logging.debug(
                    f"Model predicted {torch.argmax(output, dim=1).data}; "
                    f"Correct was {labels.data}")

        eval_loss /= len(data_loader)
        num_examples = len(data_loader.dataset)
        accuracy = float(total_correct) / num_examples

        logging.info(
            formatter.evaluation_message(
                accuracy,
                num_examples,
                epoch,
                eval_loss,
                total_correct
            )
        )
        return eval_loss, accuracy

    def _reset_hidden_state(self):
        self.model.reset_hidden_state()

    def _save_checkpoint(self, epoch, loss, best=False):
        if self.checkpoint_dir is None:
            return
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss
        }
        name = "best-model.pt" if best else f'checkpoint-epoch-{epoch + 1}.pt'
        save(checkpoint, self.checkpoint_dir / name)
