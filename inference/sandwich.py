def truncate_input(text : str, max_length : int = 1024, init_rate : int = 75):
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


model_checkpoint = "CICLAB-Comillas/BARTSumpson"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

text = """INSERT TEXT"""
truncated = truncate_input(text)
print(truncated)

print(len(tokenizer.encode(truncated)))