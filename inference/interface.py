import gradio as gr
from transformers import pipeline

def summarize(conversation: str) -> str:
    return summarizer(conversation)

# Load the model
summarizer = pipeline("summarization", model="CICLAB-Comillas/BARTSumpson")

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=15, placeholder="Coloca la conversación aquí..."),
    outputs="text",
)
demo.launch()