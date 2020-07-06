import horovod.torch as hvd

from trainer.distributed import DistributedTrainer


class HorovodTrainer(DistributedTrainer):
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
        hvd.init()
        rank = hvd.rank()
        world_size = hvd.size()

        super().__init__(
            rank=rank,
            world_size=world_size,
            model=model,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir
        )

    def _get_optimizer(self, model, lr):
        optimizer = super()._get_optimizer(model, lr)
        return hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    def train(self, epochs):
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        return super().train(epochs)
