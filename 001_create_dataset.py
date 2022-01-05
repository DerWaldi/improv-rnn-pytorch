# system imports
import os
import glob
import argparse

# additional imports
import note_seq
from tqdm.auto import tqdm
import h5py

# internal imports
import improv_rnn

def main(args):
    mxl_files = glob.glob(os.path.join(args.input, "***/**/*.mxl"))
    print("Number of MusicXML files:", len(mxl_files))

    encoder = improv_rnn.create_improv_rnn_encoder(improv_rnn.DEFAULT_MIN_NOTE, improv_rnn.DEFAULT_MAX_NOTE)

    with h5py.File(args.output, "w") as hdf:
      counter = 0
      for fn_mxl in tqdm(mxl_files):
        try:
          sequence = note_seq.musicxml_reader.musicxml_file_to_sequence_proto(fn_mxl)
          lead_sheets = improv_rnn.extract_lead_sheets(sequence)
          for lead_sheet in lead_sheets:
            try:        
              features, labels = encoder.encode(lead_sheet.chords, lead_sheet.melody)        
              group = hdf.create_group(str(counter))
              group.create_dataset("features", data=features)
              group.create_dataset("labels", data=labels)
              counter += 1
            # This happens if a melody event isgreater than max note or lower than min note => Skip this leadsheet
            except ValueError:
              pass

        # It is not uncommen for some errors to happen since there are some incompatible files
        except note_seq.MusicXMLConversionError:
          pass
        except note_seq.PolyphonicMelodyError:
          pass
        except note_seq.chords_lib.CoincidentChordsError:
          pass
        except note_seq.MultipleTimeSignatureError:
          pass

    print("Number of generated lead sheet examples:", counter)


if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Extracts features from OpenEWLD dataset and writes features and labels to h5 file.')
    parser.add_argument("--input", default="C:/Datasets/OpenEWLD/dataset", type=str, help="Path to OpenEWLD dataset.")
    parser.add_argument("--output", default="open_ewld.h5", type=str, help="Filename of h5 output file.")
    args = parser.parse_args()

    main(args)
