import logging

import random
import torch

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import BATCH_SIZE
from torch.utils.data import DataLoader

from trainer import Trainer


class RayTrainer(Trainer):
    def __init__(
            self,
            model,
            training_set,
            batch_size,
            learning_rate,
            world_size,
            seed,
            validation_set=None,
            test_set=None,
            checkpoint_dir=None,
    ):

        super().__init__(model, training_set, batch_size, learning_rate, validation_set, test_set, checkpoint_dir)
        ray.init(address='auto', redis_password='5241590000000000')

        if world_size is None:
            logging.warning("World size is not set. This may load to inconsistent results.")

        self.train_set = training_set
        self.valid_set = validation_set
        self.trainer = TorchTrainer(
            model_creator=lambda _: self.model,
            data_creator=lambda config: self._get_ray_data_loader(config),
            optimizer_creator=lambda _, __: self.optimizer,
            loss_creator=lambda _: self.loss_fn,
            num_workers=world_size,
            training_operator_cls=CustomOperator,
            add_dist_sampler=True,  # ray handles creating a distributed sampler for us
            use_gpu=False,
            use_tqdm=True,
            config={BATCH_SIZE: batch_size, "seed": seed}
        )

    def _train(self, epochs):
        best_loss = None
        for epoch in range(epochs):
            # sampler is set by ray/util/sgd/torch/distributed_torch_runner.py:168
            train_loss = self.trainer.train()['train_loss']
            self.training_history.append(train_loss)
            validation_loss = self.trainer.validate()['val_loss']
            self.validation_history.append(validation_loss)

            if best_loss is None or best_loss > validation_loss:
                logging.info(f"New best model in epoch {epoch + 1}")
                best_loss = validation_loss
                self._save_checkpoint(epoch, validation_loss, best=True)

        # self.trainer.shutdown() results in a deadlock - ray's examples don't use it either
        self.model = self.trainer.get_model()

    def _get_ray_data_loader(self, config):
        print(config[BATCH_SIZE])
        # this ensures the final batch_size equals batch_size // num_workers
        train_loader = DataLoader(self.train_set, batch_size=config[BATCH_SIZE], shuffle=True)
        valid_loader = DataLoader(self.valid_set, batch_size=config[BATCH_SIZE], shuffle=True)
        return train_loader, valid_loader


class CustomOperator(TrainingOperator):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        seed = config["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        print(f' Rank: {kwargs["world_rank"]}')

    def train_batch(self, batch, batch_info):
        *features, target = batch
        info = super().train_batch((*features, target.squeeze()),
                                   batch_info)
        return info

    def validate_batch(self, batch, batch_info):
        *features, target = batch
        info = super().validate_batch((*features, target.squeeze()),
                                      batch_info)
        return info
