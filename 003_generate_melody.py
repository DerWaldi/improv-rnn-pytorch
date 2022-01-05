# system imports
import argparse

# additional imports
import torch
import note_seq

# internal imports
import improv_rnn

def main(args):    
    # Load model from checkpoint
    loaded_checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    encoder = loaded_checkpoint['encoder']

    model = improv_rnn.ImprovRNN(encoder, 1)
    model.load_state_dict(loaded_checkpoint['model_state'])
    model.eval()

    generated_sequence, _, _ = improv_rnn.generate_sequnce(encoder, model, backing_chords=args.backing_chords, steps_per_chord=args.steps_per_chord)
    note_seq.note_sequence_to_pretty_midi(generated_sequence).write(args.output)

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Generate melody given a chord sequence.')
    parser.add_argument("--backing_chords", default="C G Am F C G F C", type=str, help="Chord sequence.")
    parser.add_argument("--steps_per_chord", default=16, type=int, help="Number of steps per chord, 4 steps = 1 quarter note duration.")
    parser.add_argument("--model", default="checkpoints/checkpoint_1000.pth", type=str, help="Path to model checkpoint.")
    parser.add_argument("--output", default="out.mid", type=str, help="Filename of output midi file.")
    args = parser.parse_args()

    main(args)