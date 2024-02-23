
import csv
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
# print (tokenize('helspf https://ams.com #innHHHbalck!... sdf'))

nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


from nltk.stem.wordnet import WordNetLemmatizer


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

import os
import gensim
from gensim import corpora

writer=csv.writer(open('../result/topics.csv','w'))
for fname in os.listdir('../final_data/cat_extended/train_all/'):
    if not fname[-4:]=='.csv':
        continue
    text_data = []
    with open(os.path.join('../final_data/cat_extended/train_all/', fname)) as f:
        reader=csv.DictReader(f)
        for line in reader:
            tokens = prepare_text_for_lda(line['text'])
            text_data.append(tokens)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]


    NUM_TOPICS = 10
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    topics = ldamodel.show_topics(num_words=4, formatted=False)
    for topic in topics:
        row=[fname]
        for kw in topic[1]:
            row.append(kw[0])
        print(row)
        writer.writerow(row)

kws={}
reader=csv.reader(open('../result/topics.csv','r'))
for row in reader:
    fname=row[0]
    if fname not in kws.keys():
        kws[fname]=[]
    for i in range(1,len(row)):
        kws[fname].append(row[i])
for kw1 in kws.keys():
    avg_ji=0
    for kw2 in kws.keys():
        if kw1==kw2:
            continue
        avg_ji+=jaccard_similarity(kws[kw1], kws[kw2])
    avg_ji/=(len(kws)-1)
    print(kw1[:-4],'\t', avg_ji)