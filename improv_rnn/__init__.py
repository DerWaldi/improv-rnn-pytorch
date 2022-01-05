import numpy as np
import torch
import note_seq

from .model import ImprovRNN, calculate_accuracies
from .data import ImprovDataset, extract_lead_sheets, create_improv_rnn_encoder, decode_melody

CHORD_SYMBOL = note_seq.NoteSequence.TextAnnotation.CHORD_SYMBOL

# Velocity at which to play chord notes when rendering chords.
CHORD_VELOCITY = 50

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84

hparams = {
    "batch_size": 128,
    "rnn_layer_sizes": [256, 256, 256],
    "dropout_keep_prob": 0.5,
    "clip_norm": 3,
    "learning_rate": 0.001,
    "num_training_steps": 20000
}


def generate_sequnce(encoder, model, backing_chords = 'C G Am F C G F C', primer_melody = [60], steps_per_chord = 16, qpm = 120, steps_per_quarter = 4, render_chords=True):
  raw_chords = backing_chords.split()
  repeated_chords = [chord for chord in raw_chords for _ in range(steps_per_chord)]
  backing_chords = note_seq.ChordProgression(repeated_chords)

  primer_melody = note_seq.Melody(primer_melody*(len(backing_chords)-1))
  
  inputs = encoder.get_inputs_batch([backing_chords], [primer_melody], full_length=True)
  inputs = np.array(inputs)

  output = None
  hn = (torch.rand(3, 1, 256), torch.rand(3, 1, 256))

  outputs = [inputs[0, 0, -encoder.num_classes:]]
  features_complete = []
  for index in range(inputs.shape[1]):
    features = inputs[:,(index):(index+1),:]

    if output is not None:
      features[0, 0, -encoder.num_classes:] = np.eye(encoder.num_classes)[np.argmax(output)]

    features_complete.append(features[0,0,:])
    features = torch.Tensor(features).float()

    output, hn = model(features, hn)
    output = output.squeeze().detach().numpy()
    outputs.append(output)
  outputs = np.array(outputs)

  melody_events = np.argmax(outputs, axis=1) - 2
  melody_events[melody_events >= 0] += DEFAULT_MIN_NOTE
  generated_melody = note_seq.Melody(melody_events)

  generated_lead_sheet = note_seq.LeadSheet(generated_melody, backing_chords)
  generated_sequence = generated_lead_sheet.to_sequence(qpm=qpm)
  
  if render_chords:
    renderer = note_seq.BasicChordRenderer(velocity=CHORD_VELOCITY)
    renderer.render(generated_sequence)

  return generated_sequence, outputs, features_complete