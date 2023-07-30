# CallSum ☎️
CallSum is a repository for summarization of phone calls. 

<!-- TABLE OF CONTENTS -->
- [CallSum ☎️](#callsum-️)
  - [About the project ℹ️](#about-the-project-ℹ️)
  - [Libraries and dependencies 📚](#libraries-and-dependencies-)
  - [Generate synthetic data samples](#generate-synthetic-data-samples)
  - [Fine tune 🎨](#fine-tune-)
    - [How it works ⚙️](#how-it-works-️)
    - [How to use ⏩](#how-to-use-)
  - [Inference ✍️](#inference-️)
    - [How to use⏩](#how-to-use)
      - [Summarize demo ℹ️](#summarize-demo-ℹ️)
      - [Summarize single files 🗒️](#summarize-single-files-️)
      - [Summarize multiple files 📦](#summarize-multiple-files-)
      - [Summarizing all files in a folder 📁](#summarizing-all-files-in-a-folder-)
  - [Developers 🔧](#developers-)


## About the project ℹ️

***CallSum*** contains explanations of the development pipeline, including: (i) preprocessing the dataset, (ii) fine tuning a BART model with that dataset and, (iii) performing inference on the resulting model.


## Libraries and dependencies 📚

 > It is recommended to use Python version 3.10.11 to avoid possible incompatibilities with dependencies and libraries

The first step is to install the required dependencies. Fortunately, the `requirements.txt` file contains all the necessary libraries to run the code without errors. 

```bash
pip install -r requirements.txt
```

## Generate synthetic data samples

To generate synthetic data samples of client conversation, we have used [SynthAI-Datasets 🤖](https://github.com/CICLAB-Comillas/SynthAI-Datasets), which is included as a **submodule** of this repo. To initialize it, just execute the following command from a GIT terminal:

```bash
git submodule init
```

Here is an example of a generated synthetic conversation using *SynthAI-Datasets*:
```
"Cliente: ¡Hola! Quería ponerme en contacto con la empresa.

Agente: ¡Hola! ¿En qué le puedo ayudar?

Cliente: Estoy llamando para reportar un problema de cambio de tarifa de luz.

Agente: Está bien, ¿puede proporcionarme su número de teléfono y dirección para seguir adelante?

Cliente: Sí, mi número de teléfono es +34 681382011 y mi dirección es Calle de la cuesta 15.

Agente: Muchas gracias. Por favor, déjeme verificar los detalles de su cuenta. ¿Tiene alguna tarifa específica en mente?

Cliente: Estoy interesado en cambiar a la tarifa diurna.

Agente: Muy bien. Por favor, permita que verifique si esa tarifa se ajusta a sus necesidades.

Agente: ¡He aquí! Hemos verificado sus detalles y hemos encontrado que la tarifa diurna es la mejor para usted. ¿Desea cambiar a esa tarifa?

Cliente: Sí, por favor.

Agente: Está hecho. ¿Hay algo más en lo que le pueda ayudar?

Cliente: No, eso es todo. Muchas gracias por la ayuda.

Agente: De nada. ¡Que tenga un buen día!
``` 

## Fine tune 🎨

The fine tuning process starts with a general model, in this case an already fine-tuned version of large-sized BART, specifically, **bart-large-cnn-samsum**, which was first fine-tuned with *CNN Daily Mail*, a large collection of text-summary pairs, and with *Samsun*, a dataset with messenger-like conversations with their summaries.

This section, explains how to fine tune that model with a dataset similar to *Samsun* but, in this case, with the conversations and summaries in Spanish.

### How it works ⚙️

The fine tuning process will save all the metrics in your Weights and Biases account during the training. At the end of the process, the resulting model will also be saved to your Hugging Face account. That's why, keys belonging to accounts on both platforms are required to run this code.

The keys need to be stored in a `secrets.json` file in the same directory as the `finetune.py` file in the following format:

```json
{
    "huggingface" : "your_huggingface_access_token",
    "wandb" : "your_wandb_api_key"
}
```

> 🚨 Important reminder: Be careful not to upload your keys. Don't worry, we have taken it into account and this file is included in the .gitignore file so that they are not uploaded in any commit.


Both datasets that are going to be loaded, the one for training and the one for evaluation, need to have the fields `Transcripcion` and `Resumen` so that the code works properly. If you need to change these fieldnames (maybe you don't have access to changing the fieldnames in the datasets) it is as easy as changing those names in the ***preprocess_function*** function.

> ℹ️ The datasets used for training are synthetic datasets generated with the OPENAI API exclusively for this task, they do not contain any sensitive information.

In this case, the training dataset contains 10.000 conversations with their summaries in spanish. The evaluation dataset contains roughly 10 conversations since it is used for metric-monitoring purposes. Both can be changed to your own datasets by modifying the ***train_datasets*** and ***eval_datasets*** variables respectively.

The Training Arguments may be changed according to your training and metric-saving preferences, to make the most of these arguments please refer to the original documentation of the `Seq2SeqTrainingArguments` class on Hugging Face.

The `wandb.init()` method contains information on how to save the data in your Weights and Biases acount, such as the project name, run id and run name, amongst others. These parameters can also be changed depending on your preferences.

After the fine tuning is finished, a wandb alert is thrown to notify the members of the wandb team. Additionally, the fine-tuned model is saved to your Hugging Face account.


### How to use ⏩
Once you have all the requirements installed, the `secrets.json` file with your keys created and the corresponding changes in the code made (if needed), you can easily run the code with the following command in a console.

```bash
python finetune.py
```

## Inference ✍️

Finally, the last step is inference, which consists of introducing conversations into the model and collecting the results. This process can be tedious, but fortunately, the GUI developed might make the things easier. 

### How to use⏩
But first you shall run the inference code:
```bash
python inference.py
```

After a few seconds, it will advise you that the service has been deployed and it's running on local URL:
http://127.0.0.1:7860. 

Just **click on it** to follow the link and a new tab with the GUI will be opened in browser. You should see something similar to this: 

TODO:Add GUI image

At the top there are 4 tabs, each of one has an unique operation mode. We recommend you trying the demo first, it shows more visually how is the model output.

> 💡 At the bottom of the page there is an accordion element, click on it and it will display a brief description of tab.

#### Summarize demo ℹ️

This one is the simplest. You only have to copy the conversation input as plain text, paste it from the clipboard in the `Transcription` Textbox (left), then click on the `Summarize` button and wait until the output appears in the `Summary` Textbox (right). 

This video shows how to do it:
https://github.com/CICLAB-Comillas/CallSum/assets/59868153/5125a9e7-c967-440e-ba64-156839c46b13

#### Summarize single files 🗒️

This second tab has 
https://github.com/CICLAB-Comillas/CallSum/assets/59868153/2138e16d-8520-4b3c-b349-10cd6d10c572

#### Summarize multiple files 📦

https://github.com/CICLAB-Comillas/CallSum/assets/59868153/b478791f-6d91-424e-a07d-5ab125091691

#### Summarizing all files in a folder 📁

> 💡 We strongly recommend using the *click-select* method to upload the input files, rather than the drag-and-drop one, as it has been found to be buggy (files stuck in *uploading...* state). 

https://github.com/CICLAB-Comillas/CallSum/assets/59868153/915e5baa-ff08-4006-b16f-361985cfe89e

## Developers 🔧

We would like to thank you for taking the time to read about this project.

If you have any suggestions💡 or doubts❔ **please do not hesitate to contact us**, kind regards from the developers:
  * [Jaime Mohedano](https://github.com/Jatme26)
  * [David Egea](https://github.com/David-Egea)
  * [Ignacio de Rodrigo](https://github.com/nachoDRT)
