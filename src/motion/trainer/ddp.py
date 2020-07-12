from torch.distributed import init_process_group, get_rank, get_world_size
from torch.nn.parallel.distributed import DistributedDataParallel

from trainer.distributed import DistributedTrainer


class DDPTrainer(DistributedTrainer):
    def __init__(
            self,
            model,
            training_set,
            batch_size,
            learning_rate,
            validation_set=None,
            test_set=None,
            checkpoint_dir=None,
            **kwargs
    ):
        init_process_group('mpi')
        model = DistributedDataParallel(model)
        rank = get_rank()
        world_size = get_world_size()

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