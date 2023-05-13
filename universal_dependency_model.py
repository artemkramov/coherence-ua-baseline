import ufal.udpipe


class UniversalDependencyModel:
    # udpipe compiled model
    model = None

    def __init__(self, path):
        # Load model by the given path
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load model by the given path: %s" % path)

    def parse(self, sentence):
        self.model.parse(sentence, self.model.DEFAULT)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def get_tokens(self, input_text):

        items = []

        sentences = self.tokenize(input_text)
        # Loop through each sentence, split them into word, perform lemmatization
        for s in sentences:

            # Parse each sentence to retrieve features
            self.tag(s)
            self.parse(s)

            # Collect all lemmas into one list
            i = 0
            words = []
            while i < len(s.words):
                if s.words[i].id != 0:
                    words.append(s.words[i].form)
                i += 1
            items.append(words)
        return items
