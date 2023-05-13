from backbone_lstm import BackboneLSTM
from elmoformanylangs import Embedder
from universal_dependency_model import UniversalDependencyModel
import numpy as np


class BaselineCoherence:

    CLIQUE_LENGTH = 3

    def __init__(self, checkpoint='model_lstm\\lstm_weights.h5', path_embedder='model_elmo'):
        self.model = BackboneLSTM(checkpoint)
        self.embedder = Embedder(path_embedder)
        self.language_processor = UniversalDependencyModel('ufal/ufal.udpipe')

    def text_to_vectors(self, text):
        return self.embedder.sents2elmo(self.language_processor.get_tokens(text))

    def evaluate_coherence_logscore(self, text):

        word_vectors = self.text_to_vectors(text)

        if len(word_vectors) < self.CLIQUE_LENGTH:
            return 0

        cliques = []
        counter = 0
        while counter < len(word_vectors) - self.CLIQUE_LENGTH + 1:
            clique = []
            for i in range(counter, counter + self.CLIQUE_LENGTH):
                clique.append(word_vectors[i])
            counter += 1
            cliques.append(clique)

        predictions = self.model.predict(cliques)
        score = np.mean(np.log(predictions))
        return score


