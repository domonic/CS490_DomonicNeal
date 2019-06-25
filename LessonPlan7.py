from bs4 import BeautifulSoup
import requests
import nltk
from nltk.util import ngrams
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import wordpunct_tokenize, ne_chunk, pos_tag
import numpy
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

'''Create input.txt file to save extracted URL'''
file = open("input.txt", "w+", encoding="utf-8")

'''Extract URL & Text'''
web_page = requests.get("https://en.wikipedia.org/wiki/Google")
web_page_soup = BeautifulSoup(web_page.text, "html.parser")

'''Write Text to file'''
file.write(web_page_soup.get_text())

'''Read file after text has been written'''
web_page_file = open("input.txt", "r", encoding="utf-8")
read_web_page_file = web_page_file.read()

'''Perform Tokenization'''
sentence_tokens = nltk.sent_tokenize(read_web_page_file)
word_tokens = nltk.word_tokenize(read_web_page_file)

for sentence_t in sentence_tokens:
    print("Sentence Token: ", sentence_t)

print()

for word_t in word_tokens:
    print("Word Token: ", word_t)

print()

'''Perform POS'''
print("Part of Speech: ", nltk.pos_tag(word_tokens))

print()

'''Perform Stemming'''

'''Porter Stemmer'''
port_stemmer = PorterStemmer()

for port_stem in word_tokens:
    print("Port Stemm: ", port_stemmer.stem(str(port_stem)))

print()

'''Lancaster Stemmer'''
lancaster_stemmer = LancasterStemmer()

for lancast_stem in word_tokens:
    print("Lancast Stemm: ", lancaster_stemmer.stem(str(lancast_stem)))

print()

'''Snowball Stemmer'''
snowball_stemmer = SnowballStemmer('english')

for snowball_stem in word_tokens:
    print("Snowball Stemm: ", snowball_stemmer.stem(str(snowball_stem)))

print()

'''Perform Lemmatization'''
lemmatizer = WordNetLemmatizer()
for lemm in word_tokens:
    print("Word Lemmatization: ", lemmatizer.lemmatize(str(lemm)))

print()

'''Perform Trigram'''
trigram = ngrams(word_tokens, 3)

for tri in trigram:
    print("Trigram: ", tri)

print()

'''Perform Named Entity Recognition'''
for sentence_t in sentence_tokens:
    ner = ne_chunk(pos_tag(wordpunct_tokenize(sentence_t)))
    print("Named Entity Recognition: ", ner)

print()
'''Change the classifier in the given code'''
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
x_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

'''KNNeighbor'''
neighbor = KNeighborsClassifier()
neighbor.fit(x_train_tfidf, twenty_train.target)

test = fetch_20newsgroups(subset='test', shuffle=True)
x_test = tfidf_Vect.transform(test.data)

predictor = neighbor.predict(x_test)

score = metrics.accuracy_score(test.target, predictor)

print("KNNeighbor Score: ", score)

'''Change the tfidf vectorizer to use bigram '''
tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

'''Multinomial Model Building'''
multinomial_bigram = MultinomialNB()
multinomial_bigram.fit(X_train_tfidf, twenty_train.target)

test_bigram = fetch_20newsgroups(subset='test', shuffle=True)
x_test_tfidf = tfidf_Vect.transform(test_bigram.data)

predictor_bigram = multinomial_bigram.predict(x_test_tfidf)

score_bigram = metrics.accuracy_score(test_bigram.target, predictor_bigram)
print("Multinomial Bigram Score: ", score_bigram)


'''Put argument stop_words=english'''
fidf_Vect = TfidfVectorizer(stop_words='english')
x_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

'''Multinomial Model Building'''
multinomial_stop = MultinomialNB()
multinomial_stop.fit(X_train_tfidf, twenty_train.target)

test_stop = fetch_20newsgroups(subset='test', shuffle=True)
x_test_tfidf = tfidf_Vect.transform(test_stop.data)

predictor_stop = multinomial_stop.predict(x_test_tfidf)

score_stop = metrics.accuracy_score(test_stop.target, predictor_stop)
print("Multinomial Stop Score: ", score_stop)






















