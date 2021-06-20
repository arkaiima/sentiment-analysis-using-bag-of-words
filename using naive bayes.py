import string
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import codecs
import numpy as np
from numpy import array
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from sklearn import   metrics
from sklearn.metrics import auc
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from string import punctuation
from sklearn.svm import SVC
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def loadnew_file(file):
    f=open(file,'r')
    text=f.read()
    f.close()
    return text

def clean_doc(doc):
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

    token=[word for word in token if len(word)>1]
    #printable = set(string.printable)
    #token=filter(lambda x: x in printable, token)
    #print(token)
    return token
def process_data(file,vocab):
    f=open(file,'r')
    lines=list()
    for row in f:
        doc=clean_doc(row)
        doc=[v for v in doc if v in vocab]
        fn=' '.join(doc)
        lines.append(fn)
    return lines
def evaluate_on_test_data(model=None):
    predictions = model.predict(Xtest)
    correct_classifications = 0
    y_test=ytest
    print("ytest",len(y_test))
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct_classifications += 1
    accuracy = 100*correct_classifications/len(y_test)
    print (confusion_matrix(y_test, predictions) )
    return [accuracy,predictions]
def miss_classification(model=None):
    y_pr=[0]*20000
    predictions = model.predict_proba(Xtest)
    for cutoff in range(1,9,1):
        for i in range(1,20000,1):
            if(predictions[i][0]<cutoff/10):
                y_pr[i]=0
            else:
                y_pr[i]=1
        cost=confusion_matrix(ytest, y_pr)[0,1]
    print (cost)

vocab_filename = 'vocab.txt'
vocab = loadnew_file(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
positive_lines = process_data('pos.csv', vocab)
positive_lines.remove('')
negative_lines = process_data('neg.csv', vocab)
negative_lines.remove('')
#print(positive_lines)
docs = positive_lines + negative_lines
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(docs)
#Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
matrix = CountVectorizer(max_features=700)
Xtrain = matrix.fit_transform(docs).toarray()
#print(Xtrain)
ytrain = array([1 for _ in range(35924)] + [0 for _ in range(35924)])
positive_lines_test=process_data('testpos.csv',vocab)
positive_lines_test.remove('')
negative_lines_test=process_data('testneg.csv',vocab)
negative_lines_test.remove('')
docs=positive_lines_test+negative_lines_test
#print((positive_lines_test))
#tokenizer.fit_on_texts(docs)
Xtest = matrix.fit_transform(docs).toarray()
ytest = array([1 for _ in range(10000)] + [0 for _ in range(10000)])
classifier = GaussianNB()
classifier.fit(Xtrain, ytrain)
miss_classification(classifier)
y_pred = classifier.predict(Xtest)
#print(y_pred)
accuracy = accuracy_score(ytest, y_pred)
print(accuracy)
print (confusion_matrix(ytest, y_pred) )
fpr, tpr, threshold = metrics.roc_curve(ytest, y_pred)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
