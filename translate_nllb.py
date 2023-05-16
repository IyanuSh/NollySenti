# https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
#from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, \
    M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import argparse
# WandB – Import the wandb library
import wandb

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


MODEL_CLASSES = {
    "mbart": (MBartForConditionalGeneration, MBart50TokenizerFast),
    "m2m100": (M2M100ForConditionalGeneration, M2M100Tokenizer),
    "nllb": (AutoModelForSeq2SeqLM, AutoTokenizer),
}


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


class PredictDataset(Dataset):

    def __init__(self, sentences, tokenizer, source_len):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.ctext = sentences  # self.data.ctext

    def __len__(self):
        return len(self.ctext)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len, pad_to_max_length=True,
                                                  return_tensors='pt', truncation=True)
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
        }

def predict(tokenizer, model_type, model, device, loader, lang_code="fr_XX"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=400,
                num_beams=5,
                forced_bos_token_id=tokenizer.lang_code_to_id[lang_code]
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            if _ % 100 == 0:
                print(f'Completed {_}')
            #print(f'Completed {_}')

            predictions.extend(preds)

    return predictions


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: e.g bart-base, bart-large, t5-base",
    )

    parser.add_argument(
        "--data_split",
        default=None,
        type=str,
        help="data split of file.",
    )

    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        required=True,
        help="The input file for sensitive information. The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--src_lang",
        default='en',
        type=str,
        required=True,
        help="source language code",
    )

    parser.add_argument(
        "--tgt_lang",
        default='fr',
        type=str,
        required=True,
        help="target language code",
    )
    parser.add_argument(
        "--max_seq_length",
        default=400,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # WandB – Initialize a new run
    wandb.init(project="transformers_translation")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config  # Initialize config
    args = parser.parse_args()
    wandb.config.update(args)  # adds all of the arguments as config variables

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.seed)  # pytorch random seed
    np.random.seed(config.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    model_class, tokenizer_class = MODEL_CLASSES[config.model_type][0], MODEL_CLASSES[config.model_type][1]
    tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    # model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model_class.from_pretrained(config.model_name_or_path)
    model = model.to(device)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    if config.do_predict:
        df = pd.read_csv(config.input_path+config.data_split+'.tsv', sep='\t')
        df.columns = ['review', 'sentiment']
        tokenizer.src_lang = "en"
        test_set = PredictDataset(df['review'], tokenizer, config.max_seq_length)

        test_params = {
            'batch_size': config.per_gpu_eval_batch_size,
            'shuffle': False,
            'num_workers': 0
        }
        test_loader = DataLoader(test_set, **test_params)
        # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
        # Saving the dataframe as predictions.csv
        print(
            'Now generating translation on our fine tuned model for the validation dataset and saving it in a dataframe')

        lang_predictions = predict(tokenizer, config.model_type, model, device, test_loader, lang_code=config.tgt_lang)

        output_dir = 'data/imdb_' + config.tgt_lang + '/'
        create_dir(output_dir)
        df[config.tgt_lang] = lang_predictions

        df[[config.tgt_lang, 'sentiment']].to_csv(output_dir + config.data_split+'.tsv', sep='\t', index=None)



        print('Output Files generated for texts')



if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=3 python3 translate_nllb.py \
--input_path data/imdb/ \
--src_lang en \
--tgt_lang fr \
--model_type m2m100 \
--model_name_or_path masakhane/m2m100_418M_en_pcm_rel_news \
--data_split train \
--max_seq_length 400 \
--seed 1 \
--do_predict


CUDA_VISIBLE_DEVICES=3 python3 translate_nllb.py \
--data_split train \
--input_path data/imdb/ \
--src_lang eng_Latn \
--tgt_lang hau_Latn \
--model_type nllb \
--model_name_or_path facebook/nllb-200-distilled-600M \
--max_seq_length  400 \
--seed 1 \
--do_predict



CUDA_VISIBLE_DEVICES=3 python3 translate_nllb.py \
--data_split dev \
--input_path data/imdb/ \
--src_lang eng_Latn \
--tgt_lang yor_Latn \
--model_type nllb \
--model_name_or_path facebook/nllb-200-distilled-600M \
--max_seq_length  400 \
--per_gpu_eval_batch_size 8 \
--seed 1 \
--do_predict



CUDA_VISIBLE_DEVICES=1 python3 translate_nllb.py \
--data_split dev \
--input_path data/imdb/ \
--src_lang eng_Latn \
--tgt_lang hau_Latn \
--model_type nllb \
--model_name_or_path facebook/nllb-200-distilled-600M \
--max_seq_length  400 \
--seed 1 \
--do_predict
'''

