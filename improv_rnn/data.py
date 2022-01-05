import copy

import numpy as np
import note_seq
import h5py

import torch


class ImprovDataset(torch.utils.data.Dataset):
    def __init__(self, fn_h5, encoder, batch_size):
        self.encoder = encoder
        self.batch_size = batch_size
        self.hdf = h5py.File(fn_h5, "r")

    def __getitem__(self, index):
        example_name = str(index)
        loaded = self.hdf[example_name]

        features = np.array(loaded["features"])
        labels = np.array(loaded["labels"])

        if self.batch_size:
            delta = self.batch_size - labels.shape[0]
            if delta > 0:
                # Pad end of sequence with a default value to match batch size
                features = np.pad(features, ((0, delta), (0, 0)))
                features[-delta:, -self.encoder.num_classes +
                         self.encoder.default_event_label] = 1
                labels = np.pad(labels, ((0, delta)),
                                constant_values=self.encoder.default_event_label)
            elif delta < 0:
                # Truncate end of sequence to match batch size
                features = features[:delta, :]
                labels = labels[:delta]

        features = torch.Tensor(features).float()
        labels = torch.Tensor(labels).long()
        return features, labels

    def __len__(self):
        return len(self.hdf.keys())


def extract_lead_sheets(sequence, steps_per_quarter=4, min_bars=7, max_steps_truncate=512, gap_bars=1.0, pad_end=False, ignore_polyphonic_notes=True, filter_drums=True, transpose_to_all_keys=True):
    """Extracts a LeadSheets from the given NoteSequence.
    https://github.com/magenta/magenta/blob/be6558f1a06984faff6d6949234f5fe9ad0ffdb5/magenta/pipelines/lead_sheet_pipelines.py#L32
    Args:
      sequence: A NoteSequence.
      steps_per_quarter: Temporal resolution in steps per quarter note.
      min_bars: Minimum length of melodies in number of bars. Shorter melodies are
          discarded.
      max_steps_truncate: Maximum number of steps in extracted melodies. If
          defined, longer melodies are truncated to this threshold. If pad_end is
          also True, melodies will be truncated to the end of the last bar below
          this threshold.
      max_steps_discard: Maximum number of steps in extracted melodies. If
          defined, longer melodies are discarded.
      gap_bars: A melody comes to an end when this number of bars (measures) of
          silence is encountered.
      min_unique_pitches: Minimum number of unique notes with octave equivalence.
          Melodies with too few unique notes are discarded.
      ignore_polyphonic_notes: If True, melodies will be extracted from
          `quantized_sequence` tracks that contain polyphony (notes start at
          the same time). If False, tracks with polyphony will be ignored.
      pad_end: If True, the end of the melody will be padded with NO_EVENTs so
          that it will end at a bar boundary.
      filter_drums: If True, notes for which `is_drum` is True will be ignored.
      transpose_to_all_keys: If True, lead sheets will be transposed into all 12 keys.
    Returns:
      lead_sheets: A list of LeadSheet instances.
    """
    lead_sheets = []
    for sequence in note_seq.sequences_lib.split_note_sequence_on_time_changes(sequence):
        quantized_sequence = note_seq.quantize_note_sequence(
            sequence, steps_per_quarter=steps_per_quarter)
        instruments = list(set(n.instrument for n in quantized_sequence.notes))
        melody = note_seq.Melody()
        melody.from_quantized_sequence(quantized_sequence, instrument=instruments[0], gap_bars=gap_bars,
                                       pad_end=pad_end, ignore_polyphonic_notes=ignore_polyphonic_notes, filter_drums=filter_drums)

        # Require a certain melody length.
        if len(melody) < melody.steps_per_bar * min_bars:
            continue

        # Truncate melodies that are too long.
        if max_steps_truncate is not None and len(melody) > max_steps_truncate:
            truncated_length = max_steps_truncate
            if pad_end:
                truncated_length -= max_steps_truncate % melody.steps_per_bar
            melody.set_length(truncated_length)

        # Extract corresponding chord progression from Sequence
        chords = note_seq.ChordProgression()
        chords.from_quantized_sequence(
            quantized_sequence, melody.start_step, melody.end_step)

        lead_sheet = note_seq.LeadSheet(melody, chords)

        if transpose_to_all_keys:
            # Transpose to all keys
            for amount in range(-6, 6):
                transposed_lead_sheet = copy.deepcopy(lead_sheet)
                transposed_lead_sheet.transpose(amount)
                lead_sheets.append(transposed_lead_sheet)
        else:
            lead_sheets.append(lead_sheet)

    return lead_sheets


def create_improv_rnn_encoder(min_note, max_note):
    """Creates an Encoder for Improv RNN containing pitch and chord information.
    https://github.com/magenta/magenta/blob/be6558f1a06984faff6d6949234f5fe9ad0ffdb5/magenta/models/improv_rnn/improv_rnn_model.py#L148
    [0]: Whether or not this chord is "no chord".
    [1, 12]: A one-hot encoding of the chord root pitch class.
    [13, 24]: Whether or not each pitch class is present in the chord.
    [25, 36]: A one-hot encoding of the chord bass pitch class.
    [37]: No event in melody.
    [38]: Note off event in melody.
    [39, ...]: note-on event for that pitch relative to the [min_note, max_note) range
    """
    return note_seq.ConditionalEventSequenceEncoderDecoder(
        note_seq.PitchChordsEncoderDecoder(),
        note_seq.OneHotEventSequenceEncoderDecoder(note_seq.MelodyOneHotEncoding(min_note=min_note, max_note=max_note)))


def decode_melody(features, min_note):
    """Convert to melodies_lib.Melody format (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch).
    Args:
      features: A 2 dimensional numpy array.
      min_note: Minimal note pitch.
    Returns:
      melody: A Melody instance.
    """
    melody_events = np.argmax(features[37:, :], axis=0) - 2
    melody_events[melody_events >= 0] += min_note
    return note_seq.Melody(melody_events)


def preprocess(encoder, batch_size=512):
    features, labels = encoder.encode(lead_sheet.chords, lead_sheet.melody)

    if batch_size:
        delta = batch_size - labels.shape[0]
        if delta > 0:
            # Pad end of sequence with a default value to match batch size
            features = np.pad(features, ((0, delta), (0, 0)))
            features[-delta:, -encoder.num_classes +
                     encoder.default_event_label] = 1
            labels = np.pad(labels, ((0, delta)),
                            constant_values=encoder.default_event_label)
        elif delta < 0:
            # Truncate end of sequence to match batch size
            features = features[:delta, :]
            labels = labels[:delta]

    features = torch.Tensor(features).float()
    labels = torch.Tensor(labels).long()
