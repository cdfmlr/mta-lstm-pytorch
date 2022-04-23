#!/usr/bin/env python
# coding: utf-8

# # Train Word2Vec model
# This notebook is aim to prepare a vector file for further usage.

# In[1]:


# get_ipython().system('pip install jieba tqdm gensim -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[2]:


import jieba
import logging
import pandas as pd
from tqdm import tqdm


# In[3]:


# get_ipython().system('pwd')


# In[4]:


file_path = ''


# In[5]:


output = open(file_path+'composition_seg.txt', 'w', encoding='utf-8')
num_lines = sum(1 for line in open(file_path+'composition.txt', 'r'))
with open(file_path+'composition.txt') as f:
    for idx, line in tqdm(enumerate(f), total=num_lines):
        if idx > 305000:
            print('\nextract %d articles' % idx)
            break
        article = line.strip('\n')
        article, topics = article.split(' </d> ')
        output.write(article)
        output.write(' \n')
    f.close()
    
output.close()


# In[6]:


word2vec_params = {
    'sg': 1,
    "vector_size": 100,
    "alpha": 0.01,
    "min_alpha": 0.0005,
    'window': 10,
    'min_count': 1,
    'seed': 1,
    "workers": 24,
    "negative": 0,
    "hs": 1,
    'compute_loss': True,
    'epochs': 50,
    'cbow_mean': 0,
}


# In[7]:


from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence(file_path+"composition_seg.txt")
model = word2vec.Word2Vec(sentences=sentences, **word2vec_params)


# In[8]:


model.save("composition_mincount_1_305000_vec_original.model")
out = file_path+'composition_mincount_1_305000_vec_original.txt'
model.wv.save_word2vec_format(out, binary=False)


# In[9]:


res = model.wv.most_similar("夏天",topn = 10)
print(f'{res=}')


# In[11]:


print(f'{len(model.wv.index_to_key)=}')


# In[ ]:




