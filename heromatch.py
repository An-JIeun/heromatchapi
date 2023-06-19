from marvel import Marvel
import pandas as pd
import tqdm
from tqdm import tqdm
import json
import  nltk
import googletrans
from googletrans import Translator
translator = googletrans.Translator()
import praw

import urllib.request
import requests
import re

import wikipediaapi
wiki=wikipediaapi.Wikipedia('en') 

from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from numpy import dot
from numpy.linalg import norm

from nltk.stem import WordNetLemmatizer
import spacy

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from PIL import Image

nlp = spacy.load('en_core_web_sm')
char = pd.read_csv("refined_dataset.csv")

def transform(vec_str):
    vec_str = vec_str.replace("[", '')
    vec_str = vec_str.replace("]", '')
    vec_str = vec_str.replace("\n", '')
    vec_str = vec_str.replace(" ", ",")
    vec_str = vec_str.split(",")
    arr_removed = [float(i) for i in vec_str if i !='']
    arr_removed = np.array(arr_removed)
    return arr_removed


def getLemma(text):
  lemmatizer = WordNetLemmatizer()
  lemm = [lemmatizer.lemmatize(w) for w in text]  
  return lemm
def sameChar(tl1,tl2):
   samepart = []
   for i in tl1:
      if i in tl2:
         samepart.append(i)
   return samepart

df= pd.read_csv("marvel_charcters.csv")
char = df[df['description'] != "NULL"].reset_index(drop=True)

def text_cleaner(txt):
    cleanr = re.compile('<.*?>')
    txt = re.sub(cleanr, '', txt)
    txt = txt.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    txt = tokenizer.tokenize(txt)
    txt = " ".join(txt)
    txt = txt.split()
    stops = set(stopwords.words("english"))
    txt = [w for w in txt if not w in stops]
    txt = " ".join(txt)
    
    return txt

# 코사인 유사도 구하는 함수
def cos_sim(me, comp):
  return dot(me, comp)/(norm(me)*norm(comp))

# getAdjVerb - 텍스트에서 형용사와 동사만 추출 
def getAdjVerb(text):
  text = nlp(text)
  n_text = []
  for token in text:
    if token.pos_ == 'ADJ' or token.pos_ == 'VERB':
      n_text.append(token.text)
  text_re = " ".join(n_text)
  return text_re

def get_document_vectors(document_list):
    word2vec_model = Word2Vec.load("word2vec.bin")
    document_embedding_list = []

    # 각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line.split():
            if word in word2vec_model.wv.vocab:
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = word2vec_model[word]
                else:
                    doc2vec = doc2vec + word2vec_model[word]

        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)

    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list

# findSimHeroModule - cosine 유사도 분석으로 본인이 작성한 텍스트와 description이 유사한 캐릭터 찾기
def findSimHeroModuleW2V(text):
   char = pd.read_csv("refined_dataset.csv")
   for i in range(len(char)):
      char['w2v_vavg'][i] = transform(char['w2v_vavg'][i])  
   result = char.copy()
   sim = []
   text = text_cleaner(text)
   
   t_arr = get_document_vectors([text])[0]

   for i in tqdm(range(len(char))):
        sim_item = cos_sim(t_arr, result['w2v_vavg'][i])
        sim.append(sim_item)
      
   result['similarity_W2V'] = sim

   result = result.sort_values(by=['similarity_W2V'], axis=0, ascending=False).reset_index(drop=True)

   return result


def promptV2(mode, txt):
    if mode == 'en':
      pass
    else:
       txt= translator.translate(txt, dest='en').text
    result = findSimHeroModuleW2V(txt)
    nm_li = result.iloc[0]
    return_val = nm_li['name']
    return_desc = translator.translate(nm_li['description'], dest='ko').text
    return return_val, return_desc


# 레딧 워드클라우드 제작 코드

user_agent = "heromatch 1.0 by /u/llempicka"
reddit = praw.Reddit(
    client_id = "ffLKNzk2Q8_SjjN_UIeg0g",
    client_secret = "U9-lcBuS8lFKiuMq10M4_vzv9qleQg",
    user_agent = user_agent 
)
mask = np.array(Image.open("reddit-logo.png")) 

# 레딧 서브레딧 'Marvel'에서 히어로 이름으로 검색한 결과를 데이터프레임으로 반환 
# 상위 5개 게시물의 타이틀, 본문, 댓글을 df로 반환
def getRedditText(heroname) :
    subreddit=reddit.subreddit("Marvel")
    resp = subreddit.search(heroname, limit=5)
    result_list = []

    for submission in tqdm(resp):
        results = {}
        results['title'] = submission.title.encode('ascii', 'ignore')
        results['score'] = submission.score
        results['text'] = submission.selftext[:120].encode('ascii', 'ignore')

        post = reddit.submission(submission.id)
        total_cmt = 0
        writelist = ""
        for top_level_comment in tqdm(post.comments):
            writelist+=top_level_comment.body.rstrip("\n")
            writelist  = writelist.replace("\n", "")
            total_cmt += 1 
        results['comments'] = writelist
        result_list.append(results)
    result_df = pd.DataFrame(result_list)
    result_df = result_df.sort_values("score", ascending = False)
    return result_df   

# wordcloud 전용 텍스트 전처리 함수
def text_cleaner_wc(txt):
    txt = txt.replace("r/", "")
    txt = txt.replace("b'", "")
    cleanr = re.compile('<.*?>')
    txt = re.sub(cleanr, '', txt)
    txt = txt.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    txt = tokenizer.tokenize(txt)
    txt = " ".join(txt)
    txt = txt.split()
    stops = set(stopwords.words("english"))
    txt = [w for w in txt if not w in stops]
    txt = " ".join(txt)
    return txt

# 워드클라우드를 생성해서 src assets 폴더에 저장장
def makeWordCloud(df, heroname):
    temp = df.copy()
    corp = ""
    for i in range(len(temp)):
        text = str(temp['title'][i]) + str(temp['text'][i]) + str(temp['comments'][i])
        text  = text_cleaner_wc(text)
        text = text.replace(heroname, "")
        corp += text
    wordcloud = WordCloud(background_color='white', width=1200, height=1200, mask=mask, colormap = 'YlOrRd').generate(corp)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('../dist/img/wordcloud.png')
