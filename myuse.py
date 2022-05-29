import tensorflow as tf
from Model.mysegmenter import mySegmenter
import numpy as np
from nltk.tokenize import sent_tokenize

class myTextSegmenter():

    def __init__(self, model_weights, max_sentences=32):
        self.model = mySegmenter(max_sentences)
        _ = self.model([['Sentence 0', 'sentence 1', 'sentence 3'], ['Sentence 0', 'sentence 1', 'sentence 3']], prepare_inputs=True)
        self.model.load_weights(model_weights)

    def segment(self, text):
        sentences = sent_tokenize(text)
        scores = self.model([sentences], prepare_inputs=True).numpy()[0][:len(sentences)]
        results = list(np.argmax(scores, axis=-1))
        passages = []
        for i, (sentence, result) in enumerate(zip(sentences, results)):
            if i == 0:
                passage = sentence
                continue
            if result == 1:
                passages.append(passage)
                passage = sentence
            else:
                passage += ' '+sentence
        passages.append(passage)
        return passages

    @staticmethod
    def print_passages(passages):
        for p in passages:
            print(p+'\n')