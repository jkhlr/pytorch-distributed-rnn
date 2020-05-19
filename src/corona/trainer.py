
from torch import save
from torch.nn import MSELoss
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, get_rank, get_world_size
from torchnet.dataset import SplitDataset


class Trainer:
    loss_fn = MSELoss()

    def __init__(self, model, training_set, checkpoint_dir=None):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.data_loader = self._get_data_loader(training_set)
        self.optimizer = self._get_optimizer(model)

    @staticmethod
    def _get_optimizer(model):
        return Adam(model.parameters(), lr=1e-3)

    @staticmethod
    def _get_data_loader(training_set):
        return DataLoader(training_set, batch_size=10, shuffle=True)

    def train(self, epochs):
        training_history = []
        for epoch in range(epochs):
            loss_sum = 0
            for train_data, train_labels in self.data_loader:
                self._reset_hidden_state()
                self.optimizer.zero_grad()

                y_pred = self.model(train_data)
                loss = self.loss_fn(y_pred.float(), train_labels)
                loss_sum += loss.item()

                loss.backward()
                self.optimizer.step()

            loss_avg = loss_sum / (len(self.data_loader) * self.data_loader.batch_size)
            self._print_loss(epoch, loss_avg)
            if epoch % 10 == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch, loss_sum)

        return self.model.eval(), training_history

    def _reset_hidden_state(self):
        self.model.reset_hidden_state()

    def _print_loss(self, epoch, loss):
        print(f'Epoch {epoch} train loss: {loss}')

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
        save(checkpoint, self.checkpoint_dir/f'checkpoint-epoch-{epoch}.pt')


class DDPTrainer(Trainer):
    def __init__(self, model, training_set, checkpoint_dir=None):
        init_process_group('mpi')
        self.rank = get_rank()
        self.world_size = get_world_size()
        super().__init__(
            model=DistributedDataParallel(model), 
            training_set=self._get_split_training_set(training_set),
            checkpoint_dir=checkpoint_dir
        )

    def _get_split_training_set(self, training_set):
        if self.world_size == 1:
            return training_set

        partitions = {
            str(i): 1 / self.world_size
            for i in range(self.world_size)
        }
        return SplitDataset(training_set, partitions, str(self.rank))

    def _reset_hidden_state(self):
        self.model.module.reset_hidden_state()
    
    def _print_loss(self, epoch, loss):
        print(f'{self.rank}: Epoch {epoch} train loss: {loss}')

    def _save_checkpoint(self, epoch, loss):
        if self.rank != 0:
            return
