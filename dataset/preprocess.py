from os import listdir
from os.path import isfile, join, splitext, basename, dirname
from typing import Tuple
import pandas as pd
import argparse
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

# Dataset dir
DATASET_DIR = dirname(__file__)

### ----------------------------------ARGS---------------------------------------------

### Arg Parser
parser = argparse.ArgumentParser()

# Input file
parser.add_argument('-i', '--input_dir', help='Transcription file path', required=True)

# Output path
parser.add_argument('-o','--output_dir', help='Output directory for preprocessed transcription files', default=DATASET_DIR)

# Unified CSV Mode (True -> 1 output file for all input files (default), False -> 1 output file for each input file)
parser.add_argument("-u","--unified_output_file", action=argparse.BooleanOptionalAction, default=True)

# Extension
parser.add_argument('-e','--extension', help='Input file extension', default=".transcription")

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
UNIFIED = args.unified_output_file
EXT = args.extension

# If not unified and output_dir is a path to csv file then it set OUTPUT_DIR to dir level
if not UNIFIED and OUTPUT_DIR.endswith(".csv"):
    OUTPUT_DIR = dirname(OUTPUT_DIR)

### ----------------------------------FUNCTIONS---------------------------------------------

def process_line(line: str) -> Tuple[str, str, int, float, float, float or None]:
    """ Process a single line of transcription file. Args:
        * `raw_line`: Raw transcribed line
        
        Returns:
        * Person
        * Transcribed line of dialogue
        * Number of dialogue line
        * Start time
        * End time
        * Quality of transcription
    """
    try:
        # Extracts each field from that line
        person, text, n_line, t1, t2, t3 = eval(line)
    except NameError:
        # Sets "nan" fields to None
        line = line.replace("nan", "None")   
         
        # Extracts each field from that line
        person, text, n_line, t1, t2, t3 = eval(line)    
    
    return person, text, n_line, t1, t2, t3

def transcription_to_df(transcription_file_path: str) -> pd.DataFrame:
    """ Converts the input file at indicated `path` to a pandas `Dataframe`. Args:
        * `transcription_file_path`: Path of input file
    """
    # Empty df with headers
    df = pd.DataFrame(columns=['person','text','line_id','t_init','t_end','Quality'])

    # Reads all lines from input file
    with open(transcription_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        serie = pd.Series(process_line(line), index=df.columns).to_frame().T
        # Appends line
        df = pd.concat([df, serie], ignore_index=True)
        
    return df

def generate_transcription(df: pd.DataFrame) -> str:
    """ Returns the transcription as a single `str` by merging all dataframe dialogue lines in one. Args: 
        * `df`: Dataframe with columns `person` and `text`.
    """
    transcription = ""
    # Concats person and text in a line
    for _, row in df.iterrows():
        transcription += f"{row['person']}: {row['text']}\n"
        
    return transcription

def transcribe_to_csv(transcription_file_path: str, csv_path: str = None) -> None:
    """ Converts a `.transcription` file to a CSV. Output columns are `id` and `transcription`. Args:
        * transcription_file_path: Path of input .transcription file
        * csv_path: Path of output CSV file  
    """
    # Converts the trasncription file to df
    transcription_df = transcription_to_df(transcription_file_path)

    csv_df = pd.DataFrame(columns=['id','transcription'])

    # Transcription id
    id = splitext(basename(transcription_file_path))[0]

    # Generates the conversation transcription from df rows
    transcripcion = generate_transcription(transcription_df)

    csv_df = pd.concat([csv_df, pd.Series([id, transcripcion], index=csv_df.columns).to_frame().T], ignore_index=True)

    # If csv path is None the output path is the same as the transcription file (same name also)
    csv_path = csv_path if csv_path is not None else transcription_file_path.replace(EXT,".csv")

    # Saves CSV
    csv_df.to_csv(csv_path, header=False, index=False, mode='a', sep=';', encoding='utf-8') if isfile(csv_path) else csv_df.to_csv(csv_path, header=True, index=False, index_label="index", sep=';', encoding='utf-8')

# Progress bar
progress_bar = Progress(
    TextColumn('[bold blue]Progreso: [bold purple]{task.percentage:.2f}% ({task.completed}/{task.total})'),
    BarColumn(),
    TimeElapsedColumn()
)

# Sub progress bar
sub_progress_bar = Progress(
    TextColumn('{task.description}'),
    SpinnerColumn('dots')
)

group = Group(
    progress_bar,
    Rule(style='#AAAAAA'),
    sub_progress_bar
)

# Autorender
live = Live(group)

if __name__ == "__main__":

    with live:

        # Single file or dir with transcription files
        transcription_files = [join(INPUT_DIR,f) for f in listdir(INPUT_DIR) if isfile(join(INPUT_DIR, f)) and f.endswith(EXT)] if not INPUT_DIR.endswith(EXT) else [INPUT_DIR]

        # Progress bar initialization
        progress_id = progress_bar.add_task(description = 'Transcription progress bar', total=len(transcription_files))

        # Progress bar initialization
        sub_progress_id = sub_progress_bar.add_task(description='[bold yellow]Transcribing')

        if UNIFIED:
            # Output path of csv file
            CSV_PATH = OUTPUT_DIR if OUTPUT_DIR.endswith(".csv") else join(OUTPUT_DIR,"transcription.csv")

            for transcription_file in transcription_files: 
                # Transcribe to CSV
                transcribe_to_csv(transcription_file,CSV_PATH)

                # Update progress bar
                progress_bar.update(progress_id, advance=1, refresh=True)

            # Save advise
            sub_progress_bar.update(sub_progress_id, description=f'[bold green] Saved {len(transcription_files)} found transcriptions at {CSV_PATH}')

        else:

            for transcription_file in transcription_files: 
                # Transcription id (name)
                id = splitext(basename(transcription_file))[0]

                # Output path -> 'id.csv'
                csv_path = join(OUTPUT_DIR,f"{id}.csv")

                # Transcribe to CSV
                transcribe_to_csv(transcription_file,csv_path)

                # Update progress bar
                progress_bar.update(progress_id, advance=1, refresh=True)

                # Update progress bar
                sub_progress_bar.update(sub_progress_id, advance=1, refresh=True, description=f'[bold green] Saved as {csv_path}')