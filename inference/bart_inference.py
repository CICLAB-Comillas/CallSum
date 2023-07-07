from transformers import pipeline
import pandas as pd

# Load the model
summarizer = pipeline("summarization", model="CICLAB-Comillas/BARTSumpson")

# Read the CSV file with the conversations to which the summary will be performed
df = pd.read_csv('datasetTEST.csv', sep = ';')

# Apply the model to the conversations
sums = []
for conv in df['Transcripcion']:
  sums.append(summarizer(conv)[0]['summary_text'])

# Join both dataframes: conversation + summary
df_sums = pd.Series(sums).to_frame(name='Resumen')
df_conv = pd.Series(df['Transcripcion'][0:44]).to_frame(name='Transcripcion')
df2 = pd.concat([df_conv, df_sums], axis = 1)

# Save the resulting dataframe as a CSV file
df2.to_csv('sum_post_model.csv')
