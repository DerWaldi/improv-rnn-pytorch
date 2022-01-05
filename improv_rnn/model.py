import numpy as np
import torch


class ImprovRNN(torch.nn.Module):
    """ ImprovRNN Model.
    https://github.com/magenta/magenta/blob/be6558f1a06984faff6d6949234f5fe9ad0ffdb5/magenta/models/shared/events_rnn_graph.py#L68
    """

    def __init__(self, encoder, batch_size=512, hidden_size=256, num_layers=3):
        super(ImprovRNN, self).__init__()

        self.encoder = encoder
        self.batch_size = batch_size

        self.rnn = torch.nn.LSTM(self.encoder.input_size,
                                 hidden_size, dropout=0.5, num_layers=num_layers, batch_first=True)
        self.logits = torch.nn.Linear(256, self.encoder.num_classes)

    def forward(self, x, h0=None):
        x, h = self.rnn(x, h0)
        x = self.logits(x)
        return x, h


def calculate_accuracies(predictions, labels, no_event):
    correct_predictions = torch.eq(labels, predictions).float()
    event_positions = torch.not_equal(labels, no_event).float()
    no_event_positions = torch.eq(labels, no_event).float()

    accuracy = torch.mean(correct_predictions)
    event_accuracy = torch.sum(
        correct_predictions * event_positions) / torch.sum(event_positions)
    no_event_accuracy = torch.sum(
        correct_predictions * no_event_positions) / torch.sum(no_event_positions)

    return accuracy, event_accuracy, no_event_accuracy
