import tensorflow as tf
import tensorflow_hub as hub

class mySegmenter(tf.keras.Model):
    def __init__(self, max_sentences):
        super(mySegmenter, self).__init__(name='mySegmenter')
        self.max_sentences = max_sentences
        self.embed = hub.KerasLayer(r'C:\Users\dhia\OneDrive\Bureau\work\TextSegmentation\5', output_shape=[512], input_shape=[], dtype=tf.string)
        self.recurrent = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, input_shape=(None, 512), return_sequences=True))
        self.classification = tf.keras.layers.Dense(2, activation='softmax')
    def call(self, inputs, prepare_inputs=False):
        if prepare_inputs:
            inputs = mySegmenter.prepare_inputs(inputs, self.max_sentences)
        x = tf.reshape(inputs, [-1])
        x = self.embed(x) # (batch size * max_sentences, 512)
        x = tf.reshape(x, [-1, self.max_sentences, 512]) # (batch size, num sentences, 512)
        x = self.recurrent(x) # (batch size, max_sentences, 256)
        x = self.classification(x)
        return x

    @staticmethod
    def prepare_inputs(inputs, max_sentences, return_tf=True, pad=''):
        x = []
        for sentences in inputs:
            x.append(sentences[:max_sentences] + [pad]*(max_sentences-len(sentences[:max_sentences])))
        
        if return_tf:
            return tf.constant(x)
        return x