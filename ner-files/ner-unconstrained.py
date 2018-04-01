from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
import string
from nltk.stem import SnowballStemmer
from tqdm import tqdm
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    word,tag,_ = sent[i]
    nextword,nexttag,_ = sent[i + 1]
    nextnextword,nextnexttag,_= sent[i + 2]
    prevword,prevtag,_ = sent[i - 1]
    prevprevword,prevprevtag,_ = sent[i - 2]
    
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
    
    stemmer = SnowballStemmer("spanish")
    
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'tag': tag,
         
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-tag': nexttag,
 
        'next-next-word': nextnextword,
        'next-next-lemma': stemmer.stem(nextnextword),
        'next-next-tag': nextnexttag,

        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-tag': prevtag,

        'prev-prev-word': prevprevword,
        'prev-prev-lemma': stemmer.stem(prevprevword),
        'prev-prev-tag': prevprevtag,

        'all-ascii': allascii,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    train_feats = []
    train_labels = []

    for sent in tqdm(train_sents):
        sent = [(" ","B-2", "B-2"),(" ","B-1", "B-1")]+sent+[(" ","E+1", "E+1"),(" ","E+2", "E+2")]
        for i in range(2,len(sent)-2):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])
    
    test_feats = []
    test_labels = []
    
    for sent in tqdm(test_sents):
        sent = [(" ","B-2", "B-2"),(" ","B-1", "B-1")]+sent+[(" ","E+1", "E+1"),(" ","E+2", "E+2")]
        for i in range(2,len(sent)-2):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])
            
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    X_test = vectorizer.transform(test_feats)
    
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(train_labels)

    # Parameters for lgbm
    parameters = {
        'application': 'multiclass',
        'num_class': len(set(train_labels)),
        'is_unbalance': 'true',
        'boosting': 'gbdt', #'dart'
        'num_leaves': 80,
        'feature_fraction': 0.7,
        'min_data_in_leaf': 500,
        'learning_rate': 0.1,
        'num_iterations': 500,
        'max_bin': 255,
        'verbosity': 1
        }
    
    lgb_train = lgb.Dataset(X_train, encoded_labels)

    model = lgb.train(parameters,
                       lgb_train,
                       num_boost_round=5000,
                       verbose_eval=False)

    encoded_y_pred = model.predict(X_test)
    y_pred = encoder.inverse_transform(np.argmax(encoded_y_pred, axis=1))


    """ Second Round"""
    begin_chain = 0
    for i in range(1,len(y_pred)):
        if(y_pred[i]!='O'):
            if(y_pred[i-1]=='O'):
                begin_chain = i
                y_pred[i]='B'+y_pred[i][1:]
            else:
                y_pred[i]='I'+y_pred[i][1:]
                if y_pred[i][1:]!=y_pred[i-1][1:]:
                    for j in range(begin_chain, i+1):
                        y_pred[i] = y_pred[i][0]+"-MISC"
                    
    
    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in test_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")






