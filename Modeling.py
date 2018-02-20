
# coding: utf-8

# In[222]:

import numpy as np
import pandas as pd

from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[223]:

data_dir = "/Users/soniamannan/Documents/DATA401/capstone/DigitalDemocracyCapstone/data/training/"
target_col = 'transition_value'


# In[224]:

training_output_filename = data_dir + "training_utterances_n_range.csv"


# In[225]:

# split dataset evenly based on labels
def split_test_train(total, stratify_col):
    transition_rows = total[total[stratify_col] != 0]
    non_transition_rows = total[total[stratify_col] == 0]
    
    # first split transitions into training/testing
    X_train1, X_test1, y_train1, y_test1 = train_test_split(transition_rows, 
                                                    transition_rows[target_col], 
                                                    test_size=0.30, random_state=42)
    
    # assert there are only transition labels in this dataframe
    assert len(X_train1[X_train1[target_col] == 0]) == 0
    assert len(X_test1[X_test1[target_col] == 0]) == 0
    
    train_len = len(X_train1) # number of non-transitions to add to training set
    test_len = len(X_test1) # number of non-transitions to add to testing set
    
    
    # next split non-transitions into training/testing
    X_train2, X_test2, y_train2, y_test2 = train_test_split(non_transition_rows, 
                                                    non_transition_rows[target_col], 
                                                    test_size=0.30, random_state=42)
    
    # pick train_len random rows from non-transition training set
    X_train2 = X_train2.sample(n = train_len, axis=0)
    
    # pick test_len random rows from non_transitions testing set
    X_test2 = X_test2.sample(n = test_len, axis=0)
    
    # assert there are no transition utterances in non-transition training and testing set
    assert len(X_train2[X_train2[target_col] != 0]) == 0
    assert len(X_test2[X_test2[target_col] != 0]) == 0
    
    # final result, concat the dataframe
    X_train_final = pd.concat([X_train1, X_train2])
    X_test_final = pd.concat([X_test1, X_test2])
    
    return X_train_final['text'], X_test_final['text'], X_train_final[target_col], X_test_final[target_col]
    


# In[226]:

train = pd.read_table(training_output_filename, sep="~")[['text', target_col]]


# In[227]:

len(np.unique(train['transition_value']))


# In[228]:

train.head()


# In[229]:

x_train, x_test, y_train, y_test = split_test_train(train, target_col)


# In[230]:

transition_rows = train[train[target_col] != 0]


# ### Assert training and testing splits are the correct dimensions
# ### After splitting, training and testing sets should each have 50% transitions and 50% non-transitions
# ### training dimensions should be 2 * 70% of the number of transitions in the data set
# ### testing dimensions should be 2 * 30% of the number of transitions in the data set

# In[231]:

assert len(x_train) == len(y_train)


# In[232]:

assert len(x_test) == len(y_test)


# In[233]:

assert len(x_train) == int(len(transition_rows) * 0.7) * 2


# In[234]:

assert len(x_test) == (len(transition_rows) * 2) - (int(len(transition_rows) * 0.7) * 2)


# In[235]:

assert len(y_train[y_train == 0]) == len(y_train[y_train != 0])


# In[236]:

assert len(y_test[y_test == 0]) == len(y_test[y_test != 0])


# In[237]:

print ("{0}% of utterances are transitions".format((sum(y_train) / len(x_train)) * 100))


# In[238]:

x_train.head()


# ### Vectorize utterances with bag of words features

# In[239]:

count_vect = CountVectorizer()
count_vect.fit(np.hstack((x_train)))
X_train_counts = count_vect.transform(x_train)
X_test_counts = count_vect.transform(x_test)


# In[240]:

assert X_train_counts.shape[1] == X_test_counts.shape[1]


# ### Pass vectorized utterances into a Naive Bayes model

# In[241]:

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)


# ### Output accuracy on testing set

# In[242]:

assert X_test_counts.shape[0] == y_test.shape[0]


# In[243]:

clf.score(X_test_counts, y_test, sample_weight=None)


# ### Look at what the wrong predictions actually are

# In[250]:

preds = clf.predict(X_test_counts)


# In[251]:

total = pd.concat([x_test, y_test], axis=1)
total.head()


# In[252]:

total['predicted'] = preds


# In[253]:

total.head()


# In[259]:

wrongs = total[total['transition_value'] != total['predicted']]
wrongs


# In[255]:

sum(preds) / len(preds)


# ### Vectorize utterances with tf-idf

# In[119]:

transformer = TfidfTransformer(smooth_idf=True)
Xtrain_tfidf = transformer.fit_transform(X_train_counts)
Xtest_tfidf = transformer.fit_transform(X_test_counts)


# In[120]:

assert Xtrain_tfidf.shape[1] == Xtest_tfidf.shape[1]


# In[121]:

clf = MultinomialNB()
clf.fit(Xtrain_tfidf, y_train)


# In[122]:

assert Xtest_tfidf.shape[0] == y_test.shape[0]


# In[123]:

clf.score(Xtest_tfidf, y_test, sample_weight=None)


# ### Vectorize utterances with n-gram features
# ### Best accuracy when combining unigram and bigrams

# In[82]:

ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
counts = ngram_vectorizer.fit(np.hstack((x_train)))


# In[83]:

len(ngram_vectorizer.get_feature_names())


# In[84]:

ngram_vectorizer.get_feature_names()[-10:]


# In[85]:

X_train_counts = counts.transform(x_train)
X_test_counts = counts.transform(x_test)


# In[86]:

assert X_train_counts.shape[1] == X_test_counts.shape[1]


# In[87]:

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)


# ### Output accuracy

# In[88]:

assert X_test_counts.shape[0] == y_test.shape[0]


# In[89]:

clf.score(X_test_counts, y_test, sample_weight=None)


# ### Use WordNet features

# In[90]:

# replace words in an utterance with their synset
def get_synset_from_text(utterance):
    for word in utterance.split():
        syn = wordnet.synsets(word)
        lemmas = set([s.lemmas()[0].name() for s in syn])
        if syn: utterance = utterance.replace(word, ' '.join(lemmas))
        
    return utterance


# ### Replace words with their synsets

# In[91]:

x_train_word_net = [get_synset_from_text(x) for x in x_train]
x_test_word_net = [get_synset_from_text(x) for x in x_test]


# In[92]:

x_train_word_net[0]


# In[93]:

x_test_word_net[0]


# ### Vectorize synsets with bag of words

# In[332]:

count_vect = CountVectorizer()
count_vect.fit(np.hstack((x_train)))
X_train_counts = count_vect.transform(x_train)
X_test_counts = count_vect.transform(x_test)


# In[333]:

assert X_train_counts.shape[1] == X_test_counts.shape[1]


# ### Train classifier

# In[334]:

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)


# ### Get accuracy

# In[335]:

assert X_test_counts.shape[0] == y_test.shape[0]


# In[336]:

clf.score(X_test_counts, y_test, sample_weight=None)


# ### Sample utterance and synset

# In[223]:

utterance = 'SB 1008 would extend the current CEQA exemption deadline for'
for word in utterance.split():
    print (word)
    syn = wordnet.synsets(word)
    lemmas = set([s.lemmas()[0].name() for s in syn])
    print (lemmas)
    utterance = utterance.replace(word, ' '.join(lemmas))
print (utterance)


# In[97]:

syn = wordnet.synsets('carry')
lemmas = set([s.lemmas()[0].name() for s in syn])
print (lemmas)


# In[ ]:



