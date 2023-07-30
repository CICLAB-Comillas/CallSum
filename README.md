# CallSum ☎️
CallSum is a repository for summarization of phone calls. 

<!-- TABLE OF CONTENTS -->
- [About the project ℹ️](#about-the-project-ℹ️)
- [Libraries and dependencies 📚](#libraries-and-dependencies-)
- [Pipeline](#pipeline-)
- [Preprocesing 🖌️](#fine-tune-)
    - [How it works ⚙️](#how-it-works-preprocessing-)
    - [How to use ⏩](#how-to-use-preprocessing-)
- [Fine tune 🎨](#fine-tune-)
    - [How it works ⚙️](#how-it-works-finetune-)
    - [How to use ⏩](#how-to-use-finetune-)
- [Inference 🖼️](#inference-)
    - [How it works ⚙️](#how-it-works-inference-)
    - [How to use ⏩](#how-to-use-inference-)
- [Developers 🔧](#developers-)


# About the project ℹ️

***CallSum*** contains explanations of the development pipeline, including: (i) preprocessing the dataset, (ii) fine tuning a BART model with that dataset and, (iii) performing inference on the resulting model.


# Libraries and dependencies 📚

 > It is recommended to use Python version 3.10.11 to avoid possible incompatibilities with dependencies and libraries

The first step is to install the required dependencies. Fortunately, the `requirements.txt` file contains all the necessary libraries to run the code without errors. 

```bash
pip install -r requirements.txt
```


# Pipeline


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


Both datasets that are going to be loaded, the one for training and the one for evaluation, need to have the fields 'Transcripcion' and 'Resumen' so that the code works properly. If you need to change these fieldnames (maybe you don't have access to changing the fieldnames in the datasets) it is as easy as changing those names in the ***preprocess_function*** function.

> ℹ️ The datasets used for training are synthetic datasets generated with the OPENAI API exclusively for this task, they do not contain any sensitive information.

In this case, the training dataset contains 10.000 conversations with their summaries in spanish. The evaluation dataset contains roughly 10 conversations since it is used for metric-monitoring purposes. Both can be changed to your own datasets by modifying the ***train_datasets*** and ***eval_datasets*** variables respectively.

The Training Arguments may be changed according to your training and metric-saving preferences, to make the most of these arguments please refer to the original documentation of the `Seq2SeqTrainingArguments` class on Hugging Face.

The `wandb.init()` method contains information on how to save the data in your Weights and Biases acount, such as the project name, run id and run name, amongst others. These parameters can also be changed depending on your preferences.

After the fine tuning is finished, a wandb alert is thrown to notify the members of the wandb team. Additionally, the fine-tuned model is saved to your Hugging Face account.


### How to use ⏩
Once you have all the requirements installed, the `secrets.json` file with your keys created and the corresponding changes in the code made (if needed), you can easily run the code with the following command in a console.

```console
python finetune.py
```

# Developers 🔧

We would like to thank you for taking the time to read about this project.

If you have any suggestions💡 or doubts❔ **please do not hesitate to contact us**, kind regards from the developers:
  * [Jaime Mohedano](https://github.com/Jatme26)
  * [David Egea](https://github.com/David-Egea)
  * [Ignacio de Rodrigo](https://github.com/nachoDRT)
