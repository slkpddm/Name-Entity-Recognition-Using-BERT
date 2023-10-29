import os
from src.nerBERT.logging import logger
from transformers import BertTokenizerFast
from datasets import load_dataset, load_from_disk
from nerBERT.entity import *


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_name)

    def tokenize_and_align_labels(self,examples, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            # word_ids() => Return a list mapping the tokens
            # to their actual word in the initial sentence.
            # It Returns a list indicating the word corresponding to each token.
            previous_word_idx = None
            label_ids = []
            # Special tokens like `` and `<\s>` are originally mapped to None
            # We need to set the label to -100 so they are automatically ignored in the loss function.
            for word_idx in word_ids:
                if word_idx is None:
                   # set –100 as the label for these special tokens
                   label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                elif word_idx != previous_word_idx:
                   # if current word_idx is != prev then its the most regular case
                   # and add the corresponding token
                   label_ids.append(label[word_idx])
                else:
                   # to take care of sub-words which have the same word_idx
                   # set -100 as well for them, but only if label_all_tokens == False
                   label_ids.append(label[word_idx] if label_all_tokens else -100)
                   # mask the subword representations after the first subword

                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    

    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        dataset_pt = dataset.map(self.tokenize_and_align_labels, batched = True)
        dataset_pt.save_to_disk(os.path.join(self.config.root_dir,"dataset_transformation"))