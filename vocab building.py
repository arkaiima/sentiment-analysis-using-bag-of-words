from collections import Counter
from os import listdir
import re
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer
def load_newdoc(fileload):
	file = open(fileload, 'r')
	text = file.read()
	file.close()
	return text


def clean_newdoc(doc):
    token=doc.split()
    dic=str.maketrans('','',punctuation)
    token=[u.translate(dic) for u in token]
    for word in token:
         word=word.lower()
    for word in token:
         word=re.sub('[^A-Za-z]', ' ', word)
         #print(word)
         #print("arka")
    token=[word for word in token if word.isalpha()]
    stopword=set(stopwords.words('english'))
    token=[v for v in token if not v in stopword]
    stemmer = PorterStemmer()
    for word in token:
         word=stemmer.stem(word)
    return token


def process_newdocs(file, vocab):
	document = load_newdoc(file)
	tokens = clean_newdoc(document)
	vocab.update(tokens)

def save_file(phrase, file):
	data = '\n'.join(phrase)
	filenew = open(file, 'w')
	filenew.write(data)
	filenew.close()

vocab = Counter()
process_newdocs('pos.csv', vocab)
process_newdocs('neg.csv', vocab)
minno_occuranence = 1
tokens = [k for k,c in vocab.items() if c >= minno_occuranence]
save_file(tokens, 'vocab.txt')