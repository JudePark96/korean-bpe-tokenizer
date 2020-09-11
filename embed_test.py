from gensim.models import FastText, Word2Vec


w2v_path = './data/w2v-100d-sg.model'
model = Word2Vec.load(w2v_path)

print(model.wv.vocab.keys())
print(model.wv.most_similar('▁희망', topn=5))
