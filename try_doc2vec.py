# -*- coding: utf-8 -*-
# @File    : bug_fixed_prediction/try_doc2vec.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/10
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts

print(common_texts)
print(len(common_texts))
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
print(documents)
print(len(documents))

# model = Doc2Vec(documents, vector_size=3, window=2, min_count=1, workers=4)
model = Doc2Vec.load("ckpt/my_doc2vec_model")
print(model.corpus_count, model.corpus_total_words)

# model.train(documents, epochs=10, total_examples=model.corpus_count)
# model.save("ckpt/my_doc2vec_model")

vector = model.infer_vector(doc_words=['human', 'interface', 'computer'])
print(vector)
print(model.docvecs.similarity(documents[0].tags[0], documents[4].tags[0]))
print(model.docvecs.most_similar([vector], topn=3))
