import gradio as gr
from transformers import pipeline, AutoTokenizer
from os.path import splitext, dirname, basename
import pandas as pd
from typing import List

def truncate_input(text : str, max_length : int = 1024, init_rate : int = 75) -> str:
    """ Truncates the input text to the max_length specified following the initial ratio. Args:
            * `text`: Text to truncate.
            * `max_length`: Maximum input token length of the model
            * `init_rate`: Initial rate. An init_rate of 75 means that 75% of the max_length number of tokens will be used for the initial part of the text and 25% for the final part 
    """

    tokenized_input = tokenizer.encode(text)
    if len(tokenized_input) > max_length: # Triggers a warning stating that len(tokenized_input) > max_model_length
        init = max_length * init_rate // 100
        end = max_length * (100-init_rate) // 100

        input_init = tokenized_input[0:init]
        input_end = tokenized_input[-end:]
        #out = input_init + input_end #Not used, can be useful to show how sentences are cut off in the middle

        detokenized_output_init = tokenizer.decode(input_init, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        index_init = detokenized_output_init.rfind('.')
        output_post_init = detokenized_output_init[0:index_init+1] # +1 to include the period

        detokenized_output_end = tokenizer.decode(input_end, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        index_end = detokenized_output_end.find('.')
        output_post_end = detokenized_output_end[index_end+1:] # +1 not to include the period

        output_post = output_post_init + output_post_end
    else:
            output_post = text

    return output_post

def summarize(transcription: str) -> str:
    """Generates a summary from the input transcription. Args:
        * transcription: Call transcription as a string
    """

    truncated_transcription = truncate_input(transcription) # TODO: Find a way to solve this limitation

    # Summary of the transcription
    result = summarizer(truncated_transcription)
    summary = result[0]['summary_text']
    
    return summary

def get_file_extension(filename: str) -> str:
    """ Returns the extension of the input file. Args:
        * filename: Filename
    """
    return splitext(filename)[1].strip().lower()

def get_transcription_from_transcription_file(filepath: str) -> str:
    """ Gets the transcription as a single single from the .transcription file. Args:
        * filepath: Path of .transcription file
    """
    
    # Empty df with headers
    df = pd.DataFrame(columns=['person','text','line_id','t_init','t_end','Quality'])

    # Reads all lines from input file
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        try:
            # Extracts each field from that line
            person, text, n_line, t1, t2, t3 = eval(line)

        except NameError:
            # Sets "nan" fields to None
            line = line.replace("nan", "None")   
            
            # Extracts each field from that line
            person, text, n_line, t1, t2, t3 = eval(line) 

        # Appends line
        serie = pd.Series((person, text, n_line, t1, t2, t3), index=df.columns).to_frame().T
        df = pd.concat([df, serie], ignore_index=True)

    # Combines all lines to compose the transcription
    transcription = ""
    # Concats person an text in a line
    for _, row in df.iterrows():
        transcription += f"{row['person']}: {row['text']}\n"
        
    return transcription

def get_transcription_from_file(file: gr.File) -> str:
    """ Extracts the transcription transcription from the input file (.csv or .transcription). Args:
        * `file`: Gradio File object
    """

    # Get file extension
    file_ext = get_file_extension(file.name)

    match file_ext:

        case ".transcription":
            
            transcription = get_transcription_from_transcription_file(file.name)

        case ".csv":
        
            df = pd.read_csv(file.name, sep=";")
            transcription = df.iloc[-1]["transcription"]

    return transcription

def summarize_file(file: gr.File) -> str:
    """ Summarizes the transcription contained in the input file. Args:
        `file`: Either .csv or .transcription
    """

    # Gets the transcription from input file
    transcription = get_transcription_from_file(file)

    # Summarizes it
    summary = summarize(transcription)

    return summary

def summarize_multiple(dir: List[gr.File], output_csv: str = "output.csv") -> str:
    """ Summarizes the transcriptions contained in all the files at input directory. Args:
        * `dir`: list of Gradio files
        * `output_csv`: name of output .csv file
        Saves all the generated summaries in a CSV (`output.csv`)
    """
    # Empty df with headers
    df = pd.DataFrame(columns=['id','transcription','summary'])

    if dir:
        total = len(dir) # Total files
        summarized = 0 # Files summarized

        bad_files = []

        for file in dir:
            try:
                # File identifier (file name)
                id = basename(file.name).split(".")[0]

                # Gets the transcription from input file
                transcription = get_transcription_from_file(file)

                # Summarizes it
                summary = summarize(transcription)

                # Appends new line to df
                serie = pd.Series((id,transcription,summary), index=df.columns).to_frame().T
                df = pd.concat([df, serie], ignore_index=True)

                summarized += 1 # Updates summarized count
            
            except:
                # File type is not .csv or .transcription
                bad_files.append(basename(file.name))

        # Completion message
        completion = f"Finished! {(100*(summarized/total)):.2f}% summarized files ({summarized}/{total})"
        if summarized != total:
            completion += f"\n\nFound {total-summarized} file(s) that could not be summarized:\n{bad_files}"
            
        # Saves the CSV
        df.to_csv("output.csv", header=True, index=True, index_label="index", sep=';', encoding='utf-8')

        return completion, "output.csv"
    
    else:
        return f"No files selected! Place a folder at the 'Input folder box'", None

MODEL = "CICLAB-Comillas/BARTSumpson"

# Load the summarizer from Hugging Face
summarizer = pipeline("summarization", model=MODEL)

# Load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Summarization App for Customer Service Conversations ☎️")
    with gr.Tab("Demo ℹ"):
        with gr.Row():
            transcription=gr.Textbox(lines=25, label="Transcription", placeholder="Paste the transcription here...")
            summary = gr.Textbox(lines=25, label="Summary", placeholder="The summary will appear here...")
        demo_btn = gr.Button("Summarize")
        demo_btn.click(fn=summarize, inputs=transcription, outputs=summary)
    with gr.Tab("Single file 🗒️"):
        with gr.Row():
            file = gr.File(label="Input file", file_count="single",file_types=[".transcription",".csv"])
            summary = gr.Textbox(label="Summary", placeholder="The summary will appear here...")
        file_button = gr.Button("Summarize")
        file_button.click(fn=summarize_file, inputs=file, outputs=summary)
    with gr.Tab("Multiple files 📦"):
        with gr.Row():
            dir = gr.File(label="Input files", file_count="multiple", file_types=[".transcription",".csv"])
            with gr.Column():
                completion = gr.Textbox(label="Completion", placeholder="Place the files at the 'Input files' box")
                output_file = gr.File(label="Output file",file_count="single")
        directory_button = gr.Button("Summarize all")
        directory_button.click(fn=summarize_multiple, inputs=dir, outputs=[completion,output_file])
    with gr.Tab("Folder 📁"):
        with gr.Row():
            dir = gr.File(label="Input folder", file_count="directory")
            with gr.Column():
                completion = gr.Textbox(label="Completion", placeholder="Place a folder at the 'Input folder' box")
                output_file = gr.File(label="Output file",file_count="single")
        directory_button = gr.Button("Summarize all")
        directory_button.click(fn=summarize_multiple, inputs=dir, outputs=[completion,output_file])

    with gr.Accordion("💡 For more details click here!", open=False):
        gr.Markdown("1. In the first tab you can try a demo to summarize a conversation by simply pasting the transcript from the clipboard.\n\
                    2. The second tab allows you to summarize the content of a conversation from a .transcription or .csv file.\n\
                    3. In the third tab you can upload multiple files and generate summaries for each one of them at the same time. In addition, it saves the result in a CSV file that you can download directly from the interface.\n\
                    4. The fourth and last window works similarly to the previous one, only that it allows you to drag a folder instead of several files separately.")

demo.launch()