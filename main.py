#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

N_SENTENCE = "I like you"
P_SENTENCE = "I love you"

def is_positive(list_of_results):
    if list_of_results[0] > list_of_results[1]:
        return "negative"
    else:
        return "positive"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    sent_list = [N_SENTENCE, P_SENTENCE]

    for sentence in sent_list:
        tok_sent = tokenizer(sentence, return_tensors="pt")
        gen_tensor = model(**tok_sent).logits
        result = torch.softmax(gen_tensor, dim=1).tolist()[0]

        print("Sentence '{}' is {} with score: {}".format(sentence, is_positive(result), max(result[0], result[1])))


if __name__ == "__main__":
    main()
