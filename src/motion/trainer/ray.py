import logging

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from torch.utils.data.distributed import DistributedSampler

from trainer import Trainer


class RayTrainer(Trainer):
    def __init__(
            self,
            model,
            training_set,
            batch_size,
            learning_rate,
            validation_set=None,
            test_set=None,
            checkpoint_dir=None
    ):
        # TODO: change
        ray.init(address='auto', redis_password='5241590000000000')

        super().__init__(
            model=model,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            # TODO: change
            sampler=DistributedSampler(
                training_set,
                num_replicas=1,
                rank=0
            )
        )

    def _train(self, epochs):
        class operator(TrainingOperator):
            def train_batch(self, batch, batch_info):
                *features, target = batch
                print(sum(feature.sum() for feature in features))
                info = super().train_batch((*features, target.squeeze()),
                                           batch_info)
                print(info)
                return info

            def validate_batch(self, batch, batch_info):
                *features, target = batch
                info = super().validate_batch((*features, target.squeeze()),
                                              batch_info)
                return info

        formatter = self._get_formatter(epochs)

        trainer = TorchTrainer(
            model_creator=lambda _: self.model,
            data_creator=lambda _: (
                self.train_loader, self.validation_loader),
            optimizer_creator=lambda _, __: self.optimizer,
            loss_creator=lambda _: self.loss_fn,
            # TODO: change
            num_workers=1,
            training_operator_cls=operator,
            use_gpu=False
        )
        best_loss = None

        for epoch in range(epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            logging.info(formatter.epoch_start_message(epoch))
            train_loss = trainer.train()['train_loss']
            self.training_history.append(train_loss)
            validation_loss = trainer.validate()['val_loss']
            self.validation_history.append(validation_loss)

            if best_loss is None or best_loss > validation_loss:
                logging.info(f"New best model in epoch {epoch + 1}")
                best_loss = validation_loss
                self._save_checkpoint(epoch, validation_loss, best=True)

        trainer.shutdown()
