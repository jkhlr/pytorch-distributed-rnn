class TrainingMessageFormatter:
    def __init__(self, num_epochs, rank=0):
        self.rank = rank
        self.num_epochs = num_epochs

    def epoch_start_message(self, epoch):
        return 'Rank: {:02d}   Start Epoch {}'.format(self.rank, epoch)

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

    def performance_message(self, memory, duration):
        return f'{self.rank}: Memory Usage: {memory}, Training Duration: {duration}'


def percentage(current, overall):
    return 100. * (current / overall)
