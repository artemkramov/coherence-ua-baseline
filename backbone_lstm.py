import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM
import numpy as np


class BackboneLSTM:
    MAX_WORDS = 40
    EMBEDDING_SIZE = 1024

    def __init__(self, checkpoint):

        input1 = tf.keras.Input(shape=(self.MAX_WORDS, self.EMBEDDING_SIZE))
        input2 = tf.keras.Input(shape=(self.MAX_WORDS, self.EMBEDDING_SIZE))
        input3 = tf.keras.Input(shape=(self.MAX_WORDS, self.EMBEDDING_SIZE))

        input_shared = tf.keras.Input(shape=(self.MAX_WORDS, self.EMBEDDING_SIZE))

        masking_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(self.MAX_WORDS, self.EMBEDDING_SIZE))(
            input_shared)
        bilstm_layer = LSTM(1024, activation='tanh', recurrent_activation="sigmoid",
                            input_shape=(None, self.EMBEDDING_SIZE),
                            dtype=tf.float32)(
            masking_layer)

        sentence_model = Model(inputs=input_shared, outputs=bilstm_layer)

        enc1 = sentence_model(input1)
        enc2 = sentence_model(input2)
        enc3 = sentence_model(input3)

        concat_vector = tf.keras.layers.concatenate([enc1, enc2, enc3])
        dense1 = Dense(512, activation='relu')(concat_vector)
        # dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(256, activation='relu')(dense1)
        # dropout2 = Dropout(0.3)(dense2)
        dense3 = Dense(1, activation='sigmoid')(dense2)

        self.lstm_model = Model(inputs={'input1': input1, 'input2': input2, 'input3': input3}, outputs=dense3)

        self.lstm_model.load_weights(checkpoint)

    def mask_sample(self, sample):
        return tf.keras.preprocessing.sequence.pad_sequences(sample, self.MAX_WORDS, dtype='float32')

    @staticmethod
    def prepare_input(sample):
        return {'input1': sample[0], 'input2': sample[1], 'input3': sample[2]}

    def predict(self, cliques):

        samples = {'input1': [], 'input2': [], 'input3': []}
        for clique in cliques:

            item = self.prepare_input(self.mask_sample(clique))

            samples['input1'].append(item['input1'])
            samples['input2'].append(item['input2'])
            samples['input3'].append(item['input3'])

        samples['input1'] = np.array(samples['input1'])
        samples['input2'] = np.array(samples['input2'])
        samples['input3'] = np.array(samples['input3'])

        predictions = self.lstm_model.predict_on_batch(samples)
        return predictions
    