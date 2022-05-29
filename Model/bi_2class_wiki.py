import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class Segmenter_bi2w(tf.keras.Model):
    def __init__(self):
        super(Segmenter_bi2w, self).__init__(name='Segmenter_bi2w')
        self.embed = hub.KerasLayer(r'C:\Users\dhia\OneDrive\Bureau\work\TextSegmentation\5', output_shape=[512], input_shape=[], dtype=tf.string)
        self.recurrent = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, input_shape=(None, 512), return_sequences=True))
        self.classification_layers = []
        for i in range(1):
            self.classification_layers.append(tf.keras.layers.Dense(128, activation='relu'))
        self.classification_layers.append(tf.keras.layers.Dense(2, activation='softmax'))
    def call(self, inputs, prepare_inputs=False):
        if prepare_inputs:
            inputs = Segmenter_bi2w.prepare_inputs(inputs,64)
        x = tf.reshape(inputs, [-1])
        x = self.embed(x) # (batch size * max_sentences, 512)
        x = tf.reshape(x, [-1, 64, 512]) # (batch size, num sentences, 512)
        x = self.recurrent(x) # (batch size, max_sentences, 256)
        for classification_layer in self.classification_layers:
                x = classification_layer(x)
        return x
    @staticmethod
    def prepare_inputs(inputs, return_tf=True, pad=''):
        x = []
        for sentences in inputs:
            x.append(sentences[:64] + [pad]*(64-len(sentences[:64])))
        
        if return_tf:
            return tf.constant(x)
        return x