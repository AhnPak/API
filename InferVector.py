from konlpy.tag import Mecab
def infer_vector(text, model):
    mecab = Mecab()
    corpus_list = []

    for string in text:
        corpus_list.extend(mecab.morphs(string))
    return model.infer_vector(corpus_list)