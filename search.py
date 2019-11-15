import re
import string
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchText:

    def run(self, text):

        similar_files = []
        for file in self.get_files():
            f = open(file, "r")
            similarity = self.process(text, f.read())
            if similarity >= 0.8:
                similar_files.append((str(file), similarity))

        return similar_files

    def process(self, text_primary, text_secundary):
        corpus = [text_primary, text_secundary]
        return self.get_cosine_similarity(corpus)

    def sanitize_text(self, text):
        pattern = "[{}]".format(string.punctuation)
        text = [word.lower() for word in text]
        text = [[re.sub(pattern, "", word) for word in words.split()]
                for words in text]
        text = [[word for word in words if len(word) > 1] for words in text]
        text = [' '.join(words) for words in text]
        return np.array(text)

    def get_cosine_similarity(self, corpus):
        set_simillarity = 0
        vectorizer = TfidfVectorizer()
        corpus = self.sanitize_text(corpus)
        features = vectorizer.fit_transform(corpus)
        simillarity = [[cosine_similarity(features[0], feature)[0, 0], int(
            i)] for i, feature in enumerate(features)]
        simillarity = np.array(
            sorted(simillarity, key=lambda sim: sim[0], reverse=True))
        result = [[corpus[y], simillarity[x, 0]]
                  for x, y in enumerate(np.int0(simillarity[1:, 1]), 1)][:1]
        for t, s in result:
            set_simillarity = round(s, 2)

        return set_simillarity

    def get_files(self):
        list_files = []
        path = Path('repo_files')
        for filepath in path.glob('*'):
            list_files.append(filepath)

        return list_files


if __name__ == "__main__":
    search = SearchText()
    text = input('Digite um texto para a busca:')
    execute = search.run(text)
    if execute:
        for file, similarity in search.run(text):
            print(
                "########################### RESULTADO DA BUSCA ###########################")
            print("Arquivo: {:16s} com {:.1%} similaridade.".format(
                file, similarity))
            print(
                "##########################################################################")
    else:
        print("Nenhum arquivo encontrado com 80% ou mais de similaridade")
