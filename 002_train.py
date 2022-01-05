# system imports
import argparse
from datetime import datetime

# additional imports
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

# internal imports
import improv_rnn

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Hyper Parameters
hparams = {
    "dataset": "open_ewld.h5",
    "batch_size": 128,
    "seq_len": 64,
    "clip_norm": 3,
    "dropout_keep_prob": 0.5,
    "learning_rate": 0.001,
    "hidden_size": 256,
    "num_layers": 3,
}

# Load Dataset
encoder = improv_rnn.create_improv_rnn_encoder(
    improv_rnn.DEFAULT_MIN_NOTE, improv_rnn.DEFAULT_MAX_NOTE
)

dataset_full = improv_rnn.ImprovDataset(hparams["dataset"], encoder, hparams["seq_len"])
train_size = int(0.8 * len(dataset_full))
test_size = len(dataset_full) - train_size
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, [train_size, test_size]
)

loader_train = torch.utils.data.DataLoader(
    dataset_train, shuffle=True, batch_size=hparams["batch_size"]
)
loader_test = torch.utils.data.DataLoader(
    dataset_test, shuffle=False, batch_size=hparams["batch_size"]
)

# Create Model
model = improv_rnn.ImprovRNN(
    encoder, hparams["seq_len"], hparams["hidden_size"], hparams["num_layers"]
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

# Training Routines


def train(loader_train, model, optimizer, hparams, writer, epoch, num_epochs):
    # Train the model
    model.train()

    total = 0
    running_loss = 0.0
    running_accuracy = 0.0
    running_event_accuracy = 0.0
    running_no_event_accuracy = 0.0

    pbar = tqdm(enumerate(loader_train), total=len(loader_train))
    for batch_ndx, (features, labels) in pbar:
        features = features.to(device)
        labels = labels.to(device)

        # forward pass and loss calculation
        output, hn = model(features)
        output = output.permute(0, 2, 1)
        loss = torch.nn.functional.cross_entropy(output, labels)

        # update metrics
        predictions = torch.argmax(output.permute(0, 2, 1), axis=2)
        accuracy, event_accuracy, no_event_accuracy = improv_rnn.calculate_accuracies(
            predictions, labels, encoder.default_event_label
        )

        running_loss += loss.item()
        running_accuracy += accuracy.item()
        running_event_accuracy += event_accuracy.item()
        running_no_event_accuracy += no_event_accuracy.item()
        total += 1

        # backward pass
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["clip_norm"])

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        # print informations
        pbar.set_description(f"[Train Epoch {epoch+1}/{num_epochs}]")
        pbar.set_postfix(
            {
                "loss": running_loss / total,
                "accuracy": running_accuracy / total,
                "event_accuracy": running_event_accuracy / total,
                "no_event_accuracy": running_no_event_accuracy / total,
            }
        )

    # Write informations to tensorboard
    writer.add_scalar("Loss/Train", running_loss / total, epoch + 1)
    writer.add_scalar("Accuracy/Train", running_accuracy / total, epoch + 1)
    writer.add_scalar("EventAccuracy/Train", running_event_accuracy / total, epoch + 1)
    writer.add_scalar("NoEventAccuracy/Train", running_no_event_accuracy / total, epoch + 1)


def evaluate(loader_test, model, writer, epoch, num_epochs):
    model.eval()

    total = 0
    running_loss = 0.0
    running_accuracy = 0.0
    running_event_accuracy = 0.0
    running_no_event_accuracy = 0.0

    with torch.no_grad():
        pbar = tqdm(enumerate(loader_test), total=len(loader_test))
        for batch_ndx, (features, labels) in pbar:
            features = features.to(device)
            labels = labels.to(device)

            # forward pass and loss calculation
            output, hn = model(features)
            output = output.permute(0, 2, 1)
            loss = torch.nn.functional.cross_entropy(output, labels)

            # update metrics
            predictions = torch.argmax(output.permute(0, 2, 1), axis=2)
            (
                accuracy,
                event_accuracy,
                no_event_accuracy,
            ) = improv_rnn.calculate_accuracies(
                predictions, labels, encoder.default_event_label
            )

            running_loss += loss.item()
            running_accuracy += accuracy.item()
            running_event_accuracy += event_accuracy.item()
            running_no_event_accuracy += no_event_accuracy.item()
            total += 1

            # print informations
            pbar.set_description(f"[Eval Epoch {epoch+1}/{num_epochs}]")
            pbar.set_postfix(
                {
                    "loss": running_loss / total,
                    "accuracy": running_accuracy / total,
                    "event_accuracy": running_event_accuracy / total,
                    "no_event_accuracy": running_no_event_accuracy / total,
                }
            )

        # write informations to tensorboard
        writer.add_scalar("Loss/Eval", running_loss / total, epoch + 1)
        writer.add_scalar("Accuracy/Eval", running_accuracy / total, epoch + 1)
        writer.add_scalar("EventAccuracy/Eval", running_event_accuracy / total, epoch + 1)
        writer.add_scalar("NoEventAccuracy/Eval", running_no_event_accuracy / total, epoch + 1)


def main(args):
    num_epochs = args.num_epochs

    # initialize tensorboard summary writer
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"logs/{time_stamp}/")

    for epoch in range(num_epochs):
        train(loader_train, model, optimizer, hparams, writer, epoch, num_epochs)

        evaluate(loader_test, model, writer, epoch, num_epochs)

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                "hparams": hparams,
                "model_state": model.state_dict(),
                "encoder": encoder,
            }
            torch.save(checkpoint, f"checkpoints/checkpoint_{epoch+1}.pth")


if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument("--num_epochs", default=1000, type=int, help="Number of epochs to train.")
    args = parser.parse_args()
    main(args)
