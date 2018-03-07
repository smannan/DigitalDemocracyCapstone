
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[2]:

data_dir = "/Users/soniamannan/Documents/DATA401/capstone/DigitalDemocracyCapstone/data/"
target_col = 'transition_value'


# In[3]:

training_output_filename = data_dir + "training/training_utterances_binary.csv"


# In[4]:

training_output_binary_filename = data_dir + "training/training_utterances_binary.csv"


# In[5]:

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
    


# In[44]:

# assert training/testing split is balanced
def verify_train_test_split(train, x_train, y_train, x_test, y_test):
    transition_rows = train[train[target_col] != 0]
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) == int(len(transition_rows) * 0.7) * 2
    assert len(x_test) == (len(transition_rows) * 2) - (int(len(transition_rows) * 0.7) * 2)
    assert len(y_train[y_train == 0]) == len(y_train[y_train != 0])
    assert len(y_test[y_test == 0]) == len(y_test[y_test != 0])
    print ("{0}% of utterances are transitions".format((sum(y_train) / len(x_train)) * 100))


# In[6]:

# extract bag of words features from text for a model
def bag_of_words_features(x_train, x_test):
    count_vect = CountVectorizer()
    count_vect.fit(np.hstack((x_train)))
    X_train_counts = count_vect.transform(x_train)
    X_test_counts = count_vect.transform(x_test)
    
    assert X_train_counts.shape[1] == X_test_counts.shape[1]
    
    return X_train_counts, X_test_counts


# In[7]:

# output accuracy for a naive bayes model
# return the trained model
def model(X_train_counts, X_test_counts, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)
    
    assert X_test_counts.shape[0] == y_test.shape[0]
    
    acc = clf.score(X_test_counts, y_test, sample_weight=None)
    print("Model accuracy {0}".format(acc))
    
    return clf


# In[8]:

def remove_transcript(train, n):
    total = len(np.unique(train['video_id']))
    ids = np.unique(train['video_id'])[:n]
    rows = train[train['video_id'].isin(ids)]
    train = train[~(train['video_id'].isin(ids))]
    
    assert len(np.unique(rows['video_id'])) == n
    assert len(np.unique(train['video_id'])) == total - n
    
    return train, rows


# In[64]:

# adds the prefix CONTEXT to all utterances n before and after
# a transition phrase
def add_context(n):
    n_range = pd.read_csv(training_output_binary_filename, sep="~")

    print ("Number of original transitions {0}".format(len(n_range[n_range['transition_value'] == 1])))
    
    transition_indexes = n_range.index[n_range["transition_value"] == 1].tolist()
    
    transition_text = n_range['text']
    labels = n_range['transition_value']
    
    new_transition_indexes = []
    new_transition_text = []

    length = len(n_range)
    
    for i in transition_indexes:
        for x in range(-n, n+1):
            if (i + x >= 0 and i + x < length):
                new_transition_indexes.append(i + x)
                
                if (labels[i+x] != 1):
                    text = ' '.join(["CONTEXT-" + x for x in transition_text[i+x].split()])
                    new_transition_text.append(text)
                    
                else:
                    new_transition_text.append(transition_text[i+x])

    n_range.loc[new_transition_indexes, "transition_value"] = 1
    n_range.loc[new_transition_indexes, "text"] = new_transition_text
    
    print ("Number of new transitions indexes {0}".format(len(new_transition_indexes)))
    print ("Number of new transitions {0}".format(len(n_range[n_range['transition_value'] == 1])))

    return n_range


# In[87]:

# combine all n_range with context-appended text
# into a single utterance
def collapse_content(uncollapsed):
    accumulated_text = ""
    accumulating = False
    
    all_text = []
    transition = []
    
    for line in uncollapsed.iterrows():
        transition_value = int(line[1]['transition_value'])
        text = line[1]['text'] + " "
        
        if transition_value == 1 and accumulating:
            accumulated_text = accumulated_text + text
            
        elif transition_value == 1 and not accumulating:
            accumulating = True
            accumulated_text = accumulated_text + text
            
        elif transition_value == 0 and accumulating:
            all_text.append(accumulated_text)
            transition.append(1)
            
            all_text.append(text)
            transition.append(0)
            
            accumulating = False
            accumulated_text = ""
            
        else:
            all_text.append(text)
            transition.append(0)
            
    res = pd.DataFrame({'text':all_text, 'transition_value':transition}, columns=['text', 'transition_value'])
    return res


# ### Read in data

# In[10]:

train = pd.read_table(training_output_filename, sep="~")[['text', target_col, 'video_id']]


# In[11]:

print("Number of transitions in the dataset {0}".format(len(train[train['transition_value'] != 0])))


# In[12]:

train.head()


# ### Remove top 5 videos from dataset

# In[11]:

train, transcripts = remove_transcript(train, 5)


# In[12]:

print("Number of transitions in dataset after removing top 5 transcripts {0}"
.format(len(train[train['transition_value'] != 0])))


# ### Split into training and testing sets

# In[13]:

x_train, x_test, y_train, y_test = split_test_train(train[['text', target_col]], target_col)


# In[14]:

transition_rows = train[train[target_col] != 0]


# ### Assert training and testing splits are the correct dimensions
# ### After splitting, training and testing sets should each have 50% transitions and 50% non-transitions
# ### training dimensions should be 2 * 70% of the number of transitions in the data set
# ### testing dimensions should be 2 * 30% of the number of transitions in the data set

# In[15]:

assert len(x_train) == len(y_train)


# In[16]:

assert len(x_test) == len(y_test)


# In[17]:

assert len(x_train) == int(len(transition_rows) * 0.7) * 2


# In[18]:

assert len(x_test) == (len(transition_rows) * 2) - (int(len(transition_rows) * 0.7) * 2)


# In[19]:

assert len(y_train[y_train == 0]) == len(y_train[y_train != 0])


# In[20]:

assert len(y_test[y_test == 0]) == len(y_test[y_test != 0])


# In[21]:

print ("{0}% of utterances are transitions".format((sum(y_train) / len(x_train)) * 100))


# In[22]:

x_train.head()


# ### Vectorize utterances with bag of words features

# ### Pass vectorized utterances into a Naive Bayes model

# ### Output accuracy on testing set

# In[23]:

X_train_counts, X_test_counts = bag_of_words_features(x_train, x_test)
bag_of_words_model = model(X_train_counts, X_test_counts, y_train, y_test)


# In[24]:

def compare_predicted_to_actual(clf, X_test_counts, x_test, y_test, outfilename):
    # get predicted values
    preds = clf.predict(X_test_counts)
    
    print("% predictions that were 1's {0}\n".format(sum(preds) / len(preds)))
    
    # add predicted values to original dataframe
    total = pd.concat([x_test, y_test], axis=1)
    total['predicted'] = preds
    
    # get the incorrect predictions and write to a csv
    wrongs = total[total['transition_value'] != total['predicted']]
    wrongs.to_csv(outfilename)
    
    print ("Example of an incorrect transition\n")
    print (list(wrongs['text'])[0])
    print ("Actual {0}".format(list(wrongs['transition_value'])[0]))
    print ("Predicted {0}".format(list(wrongs['predicted'])[0]))
    
    return wrongs


# ### Look at what the wrong predictions actually are

# In[25]:

wrongs = compare_predicted_to_actual(bag_of_words_model, X_test_counts, 
 x_test, y_test, data_dir+'predictions/wrong_predictions.csv')


# ### Vectorize utterances with tf-idf

# In[26]:

def transform_tfidf(x_train, x_test):
    X_train_counts, X_test_counts = bag_of_words_features(x_train, x_test)
    
    transformer = TfidfTransformer(smooth_idf=True)
    Xtrain_tfidf = transformer.fit_transform(X_train_counts)
    Xtest_tfidf = transformer.fit_transform(X_test_counts)
    
    assert Xtrain_tfidf.shape[1] == Xtest_tfidf.shape[1]
    
    return Xtrain_tfidf, Xtest_tfidf


# In[27]:

Xtrain_tfidf, Xtest_tfidf = transform_tfidf(x_train, x_test)


# In[28]:

tf_idf_model = model(Xtrain_tfidf, Xtest_tfidf, y_train, y_test)


# ### Vectorize utterances with n-gram features
# ### Best accuracy when combining unigram and bigrams

# In[29]:

def transform_ngram(start, stop, x_train, x_test):
    ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(start, stop))
    counts = ngram_vectorizer.fit(np.hstack((x_train)))
    
    print ("Number of transformed features {0}\n"
     .format(len(ngram_vectorizer.get_feature_names())))
    
    print ("First 10 features\n{0}"
     .format('\n'.join(ngram_vectorizer.get_feature_names()[-10:])))
    
    X_train_counts = counts.transform(x_train)
    X_test_counts = counts.transform(x_test)
    
    assert X_train_counts.shape[1] == X_test_counts.shape[1]
    
    return X_train_counts, X_test_counts


# In[30]:

X_train_ngram_counts, X_test_ngram_counts = transform_ngram(1, 2, x_train, x_test)


# In[31]:

ngram_model = model(X_train_ngram_counts, X_test_ngram_counts, y_train, y_test)


# ### For utterances in a transcript, tag what the model predicts the utterance to be

# In[42]:

def predict_entire_transcript(transcripts, x_train, x_test, y_train, y_test):
    print("{0}\n".format(transcripts.head()))
    
    count_vect = CountVectorizer()
    count_vect.fit(np.hstack((x_train)))
    transcripts_test = count_vect.transform(transcripts['text'])
    label = transcripts['transition_value']
    
    X_train_counts, X_test_counts = bag_of_words_features(x_train, x_test)
    bag_of_words_model = model(X_train_counts, X_test_counts, y_train, y_test)
    
    preds = bag_of_words_model.predict(transcripts_test)
    
    assert len(preds) == transcripts_test.shape[0]
    
    return preds


# In[47]:

preds = predict_entire_transcript(transcripts, x_train, x_test, y_train, y_test)


# In[50]:

res = transcripts.copy()
res['predicted'] = preds
res['actual'] = transcripts['transition_value']
res = res.drop(['transition_value'], axis=1)
res.head()


# In[51]:

res.to_csv('/Users/soniamannan/Documents/DATA401/capstone/DigitalDemocracyCapstone/data/predictions/binary_predicted_transcript.csv')


# ### Add a context prefix to surrounding utterances
# ### Collapse the context (with prefix) and train on bag of words

# In[88]:

n_range = add_context(5)


# In[89]:

collapsed_n_range = collapse_content(n_range)
collapsed_n_range.head()


# In[90]:

transitions = collapsed_n_range[collapsed_n_range['transition_value'] != 0]
non_transitions = collapsed_n_range[collapsed_n_range['transition_value'] == 0]


# In[91]:

print ("Number of transition phrases {0}".format(len(transitions)))


# In[92]:

print ("Total number of utterances {0}".format(len(collapsed_n_range)))


# In[93]:

print ("{0}% of utterances are transitions".format((len(transitions)/len(collapsed_n_range))*100))


# In[94]:

print ("Example transition\n\n{0}".format(list(transitions['text'])[0]))


# In[95]:

print ("Example non-transitions\n\n{0}".format('\n'.join(list(non_transitions['text'])[:15])))


# ### Make a new training/testing split

# In[96]:

x_train_context, x_test_context,  y_train_context, y_test_context = split_test_train(collapsed_n_range[['text', target_col]], target_col)


# In[97]:

verify_train_test_split(collapsed_n_range, x_train_context, y_train_context, x_test_context, y_test_context)


# In[98]:

X_train_counts, X_test_counts = bag_of_words_features(x_train_context, x_test_context)
bag_of_words_model = model(X_train_counts, X_test_counts, y_train_context, y_test_context)


# ### Use WordNet features

# In[52]:

# replace words in an utterance with their synset
def get_synset_from_text(utterance):
    for word in utterance.split():
        syn = wordnet.synsets(word)
        lemmas = set([s.lemmas()[0].name() for s in syn])
        if syn: utterance = utterance.replace(word, ' '.join(lemmas))
        
    return utterance


# ### Replace words with their synsets

# In[139]:

x_train_word_net = [get_synset_from_text(x) for x in x_train]
x_test_word_net = [get_synset_from_text(x) for x in x_test]


# In[140]:

x_train_word_net[0]


# In[141]:

x_test_word_net[0]


# ### Vectorize synsets with bag of words

# In[142]:

count_vect = CountVectorizer()
count_vect.fit(np.hstack((x_train)))
X_train_counts = count_vect.transform(x_train)
X_test_counts = count_vect.transform(x_test)


# In[143]:

assert X_train_counts.shape[1] == X_test_counts.shape[1]


# ### Train classifier

# In[144]:

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)


# ### Get accuracy

# In[145]:

assert X_test_counts.shape[0] == y_test.shape[0]


# In[146]:

clf.score(X_test_counts, y_test, sample_weight=None)


# ### Sample utterance and synset

# In[28]:

utterance = 'SB 1008 would extend the current CEQA exemption deadline for'
for word in utterance.split():
    print (word)
    syn = wordnet.synsets(word)
    lemmas = set([s.lemmas()[0].name() for s in syn])
    print (lemmas)
    print (syn)
    utterance = utterance.replace(word, ' '.join(lemmas))
print (utterance)


# In[97]:

syn = wordnet.synsets('carry')
lemmas = set([s.lemmas()[0].name() for s in syn])
print (lemmas)


# In[ ]:



