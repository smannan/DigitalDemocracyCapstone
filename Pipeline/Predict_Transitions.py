import datetime
import math
import numpy as np
import pandas as pd
import pickle
import re
import sys
import textedit
from bs4 import BeautifulSoup

# Define text formatting and bill replacement logic.  
# This converts all utterances to entirely lowercase, 
# and replaces the following instances of words in an 
# utterance with the tag BILL_ID.

bill_id_pattern_1_1 = "ab[0-9]+"
bill_id_pattern_1_2 = "sb[0-9]+"
bill_id_pattern_1_3 = "aca[0-9]+"
bill_id_pattern_1_4 = "acr[0-9]+"
bill_id_pattern_1_5 = "ajr[0-9]+"
bill_id_pattern_1_6 = "ar[0-9]+"
bill_id_pattern_1_7 = "hr[0-9]+"
bill_id_pattern_1_8 = "sca[0-9]+"
bill_id_pattern_1_9 = "scr[0-9]+"
bill_id_pattern_1_10 = "sjr[0-9]+"

bill_id_pattern_2_1 = ["ab", "[0-9]+"]
bill_id_pattern_2_2 = ["sb", "[0-9]+"]
bill_id_pattern_2_3 = ["aca", "[0-9]+"]
bill_id_pattern_2_4 = ["acr", "[0-9]+"]
bill_id_pattern_2_5 = ["ajr", "[0-9]+"]
bill_id_pattern_2_6 = ["ar", "[0-9]+"]
bill_id_pattern_2_7 = ["hr", "[0-9]+"]
bill_id_pattern_2_8 = ["sca", "[0-9]+"]
bill_id_pattern_2_9 = ["scr", "[0-9]+"]
bill_id_pattern_2_10 = ["sjr", "[0-9]+"]

bill_id_pattern_3_1 = ["assembly", "bill", "[0-9]+"]
bill_id_pattern_3_2 = ["senate", "bill", "[0-9]+"]

bill_id_pattern_4_1 = ["assembly", "bill", "number", "[0-9]+"]
bill_id_pattern_4_2 = ["senate", "bill", "number", "[0-9]+"]

# model parameters
default_model_type = "NB"
default_model_filename = "nb_model.p"
default_count_vectorizer_filename = "nb_count_vect.p"
default_cleaned_raw_filename = "cleaned_transcript.csv2"

model_type = default_model_type
model_filename = default_model_filename
count_vectorizer_filename = default_count_vectorizer_filename
cleaned_raw_filename = default_cleaned_raw_filename

THRESHOLD_PROBABILITY = .5

def re_match_lists_helper(pattern_list, word_list):
    for p in range(len(pattern_list)):
        if not (re.match(pattern_list[p], word_list[p])):
            return False
    return True

def re_match_lists(pattern_list_list, word_list):
    # for each pattern
    for pl in range(len(pattern_list_list)):
        # match the specific pattern against the word
        if (re_match_lists_helper(pattern_list_list[pl], word_list)):
            return True
    return False

def re_find_lists(pattern_list_list, word):
    res = []

    for i in range(len(pattern_list_list)):
        #print (word, ' '.join(pattern_list_list[i]))

        p = re.compile(' '.join(pattern_list_list[i]))

        res += p.findall(word)

    if len(res) == 0:
        return ''

    return ' '.join(res)

def find_any_pattern(word, pattern_list):
    return re_find_lists(pattern_list, word)

def matches_any_4_word_pattern(word1, word2, word3, word4):
    pattern_list_list = [bill_id_pattern_4_1, bill_id_pattern_4_2]
    word_list = [word1, word2, word3, word4]
    
    return re_match_lists(pattern_list_list, word_list)

def matches_any_3_word_pattern(word1, word2, word3):
    pattern_list_list = [bill_id_pattern_3_1, bill_id_pattern_3_2]
    word_list = [word1, word2, word3]
    
    return re_match_lists(pattern_list_list, word_list)
    
def matches_any_2_word_pattern(word1, word2):
    pattern_list_list = [bill_id_pattern_2_1, bill_id_pattern_2_2,
                         bill_id_pattern_2_3, bill_id_pattern_2_4,
                         bill_id_pattern_2_5, bill_id_pattern_2_6,
                         bill_id_pattern_2_7, bill_id_pattern_2_8,
                         bill_id_pattern_2_9, bill_id_pattern_2_10]
    word_list = [word1, word2]
    
    return re_match_lists(pattern_list_list, word_list)

def matches_any_1_word_pattern(word):
    return (re.match(bill_id_pattern_1_1, word) or
            re.match(bill_id_pattern_1_2, word) or
            re.match(bill_id_pattern_1_3, word) or
            re.match(bill_id_pattern_1_4, word) or
            re.match(bill_id_pattern_1_5, word) or
            re.match(bill_id_pattern_1_6, word) or
            re.match(bill_id_pattern_1_7, word) or
            re.match(bill_id_pattern_1_8, word) or
            re.match(bill_id_pattern_1_9, word) or
            re.match(bill_id_pattern_1_10, word))

def shift_words_over(words, word_ix, shift_amount):
    words_length = len(words)
    
    for i in range(word_ix, words_length - shift_amount):
        words[i] = words[i+shift_amount]
    while(len(words) > (words_length-shift_amount)):
        del words[-1]
        
    return words

# returns all the matched bill names found in utterance
def find_bill_names(utterance):
    input_list = utterance.lower().split()
    pattern_list_list = []
    res = []

    for n in range(1,5):
        ngrams = (list(zip(*[input_list[i:] for i in range(n)])))

        if n == 4: 
            pattern_list_list = [bill_id_pattern_4_1, bill_id_pattern_4_2]

        elif n == 3:
            pattern_list_list = [bill_id_pattern_3_1, bill_id_pattern_3_2]

        elif n == 2:
            pattern_list_list = [bill_id_pattern_2_1, bill_id_pattern_2_2,
             bill_id_pattern_2_3, bill_id_pattern_2_4,
             bill_id_pattern_2_5, bill_id_pattern_2_6,
             bill_id_pattern_2_7, bill_id_pattern_2_8,
             bill_id_pattern_2_9, bill_id_pattern_2_10]

        elif n == 1:
            pattern_list_list = [bill_id_pattern_1_1, bill_id_pattern_1_2,
             bill_id_pattern_1_3, bill_id_pattern_1_4, bill_id_pattern_1_5,
             bill_id_pattern_1_6, bill_id_pattern_1_7, bill_id_pattern_1_8,
             bill_id_pattern_1_9, bill_id_pattern_1_10]
        
        for word in ngrams:
            target = ' '.join(word).strip()
            bill_names = find_any_pattern(target, pattern_list_list)

            if len(bill_names) > 0: res.append(bill_names)
    
    return res

# utterance text
def replace_bill_ids_in_utterance(utterance):
    # split text into list of words
    words = utterance.lower().split()
    utterance_length = len(words)

    word_ix = 0 # index into the utterance
    bill_id_replaced = False # if something has been replaced

    # going through each word
    while(word_ix < utterance_length):
        # match a four word pattern
        if (word_ix < (utterance_length-3) and
            matches_any_4_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2],
                                         words[word_ix+3])):

            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 3)
            utterance_length -= 3
            bill_id_replaced = True
            
        # match a three word pattern
        elif (word_ix < (utterance_length-2) and
              matches_any_3_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2])):
            
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 2)
            utterance_length -= 2
            bill_id_replaced = True
            
        # match a two word pattern
        elif (word_ix < (utterance_length-1) and
            matches_any_2_word_pattern(words[word_ix],
                                         words[word_ix+1])):
            
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 1)
            utterance_length -= 1
            bill_id_replaced = True
            
        # match a one word pattern
        elif (matches_any_1_word_pattern(words[word_ix])):
            words[word_ix] = "<BILL_ID>"
            bill_id_replaced = True
            

        word_ix += 1
            
    return (" ".join(words), bill_id_replaced)

def replace_bill_ids(old, new):
    for line in old:
        line_splits = line.lower().rstrip("\n").split("~")
        (new_text, bill_id_replaced) = replace_bill_ids_in_utterance(line_splits[2])
        new.write(line_splits[0] + "~" + line_splits[1] + "~" + new_text + "~" + line_splits[3] + "\n")

# Define add context function (prefix and postfix words in surrounding utterances).
# adds the prefix POST to all utterances n after
# adds the prefix PRE to all utterances n before
# a transition phrase
def add_context(n, cleaned_raw_bill_id_replaced_filename):
    n_range = pd.read_csv(cleaned_raw_bill_id_replaced_filename, sep="~")
    
    transition_text = n_range['text']
    new_transition_text = []

    length = len(n_range)
    
    for i in range(length):
        # get the phrases in the window
        text = ''
        for x in range(-n, n+1):
            # window is within range of the dataframe
            if (i + x >= 0 and i + x < length):
                if (x > 0):
                    text += ' '.join(["POST-" + post for post in transition_text[i+x].split()])
                if (x < 0):
                    text += ' '.join(["PRE-" + pre for pre in transition_text[i+x].split()])
                else:
                    text += ' ' + transition_text[i+x] + ' '
                    
        new_transition_text.append(text)
    
    print ("Number of new phrases {0}".format(len(new_transition_text)))
    
    n_range.drop(['text'], axis=1, inplace=True)
    n_range['text'] = new_transition_text
    
    n_range.to_csv(cleaned_raw_bill_id_replaced_filename, sep="~", index=False)

def predict_entire_transcript(transcripts, model, count_vect):
    transcripts_test = count_vect.transform(transcripts['text'])
    
    probs = model.predict_proba(transcripts_test)
    preds = [1 if p[1] > THRESHOLD_PROBABILITY else 0 for p in probs]
    
    assert len(preds) == transcripts_test.shape[0]
    
    return preds

def predict_from_naive_bayes(transcript):
    model = pickle.load(open(model_filename, "rb"))
    count_vect = pickle.load(open(count_vectorizer_filename, "rb"))

    prediction_values = predict_entire_transcript(transcript, model, count_vect)
    transcript["prediction"] = prediction_values
    predicted_transitions = transcript[transcript["prediction"]==1]
    transition_dictionary = predicted_transitions[["video_id", "start", "end", "text"]].to_dict(orient="records")
    
    return transition_dictionary

def predict_from_neural_network():
    pass

# new_transcript is a list of dictionaries
def process_raw_data(new_transcript):
    ### Determine the raw .srt file that will be processed.

    cleaned_raw_filename = "cleaned_transcript.csv"
    cleaned_raw_bill_id_replaced_filename = cleaned_raw_filename + "2"

    print("...Parsing raw transcript...")

    cleaned_raw = pd.DataFrame(new_transcript)
    cleaned_raw.sort_values(["video_id", "start"]) \
     .to_csv(cleaned_raw_filename, sep="~", index=False)

    print("...Raw transcript parsed, beginning text formatting and bill replacement...")

    with open(cleaned_raw_filename, 'r') as old:
        with open(cleaned_raw_bill_id_replaced_filename, 'w') as new:
            # consume/write headings
            h = old.readline()
            new.write(h)
            
            replace_bill_ids(old, new)

    print("...Text formatted and bills replaced, adding context...")

    add_context(5, cleaned_raw_bill_id_replaced_filename)

    print("...Context added.")

    processed_new_transcript = pd.read_csv(cleaned_raw_bill_id_replaced_filename, sep="~")
    return processed_new_transcript

def enhance_dictionary_helper(combined): 
    enhanced_dictionary = []
    
    start_list = list(combined["start"])
    end_list = list(combined["end"])
    video_id_list = list(combined["video_id"])
    original_text_list = list(combined["text_original"])
    modified_text_list = list(combined["text_modified"])
    bill_names_list = list(combined["bill_names"])
    
    found_associated_bill_id = True
    i = 0
    while (i < len(original_text_list)):
        # if the utterance is not a transition utterance, continue
        if (pd.isnull(modified_text_list[i])):
            i += 1
            continue
        
        # otherwise, if there is a bill id immediately associated with the utterance,
        # input it into the dictionary, then continue
        
        if (len(bill_names_list[i]) > 0):
            enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':bill_names_list[i][0]})
            i += 1
            continue

        #otherwise, scan all table rows until the next transition utterance looking for
        #an associated bill id
        j = i+1
        while(j < len(original_text_list) and pd.isnull(modified_text_list[j])):
            if (len(bill_names_list[j]) > 0):
                enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':bill_names_list[j][0]})
                i = j+1
                found_associated_bill_id = False
                break
            j += 1
            
        if (not found_associated_bill_id):
            found_associated_bill_id = True
            continue
        
        #at this point, the transition utterance has no associated bill ids in the window
        #so we likely want to disregard the transition as a possibility
        #for now we simply enter it with the suggested_bill field having value 'NONE'
        enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':'NONE'})
        i = j
            
    return enhanced_dictionary

def collapse_dictionary(enhanced_dictionary):
    res = []
    i = 0
    j = 0
    n = len(enhanced_dictionary)
    epsilon = 5

    while i < n:
        j = i + 1
        res.append(enhanced_dictionary[i])

        while j < n and math.fabs(enhanced_dictionary[j]['start']
         - enhanced_dictionary[i]['end']) < epsilon:
            j += 1
            i += 1

        i = j

    return res

def enhance_dictionary(original_transcript, transition_dictionary):
    #transition_dictitionary format: [{video_id, start, end, text}]
    #original transcript format new_transcript = [{'start':'1', 'end':'9', 
    #'text':'we are starting', 'video_id':'1'}]
    
    original = pd.DataFrame.from_dict(original_transcript)
    transitions = pd.DataFrame.from_dict(transition_dictionary)
    
    #combined = pd.merge(original, transitions, on=["video_id", "start", "end"], suffixes=("_original", "_modified"))
    combined = pd.merge(original, transitions, 
     how='left', on=["video_id", "start", "end"], suffixes=("_original", "_modified")) 
    substituted_original = textedit.correct_text(list(combined["text_original"]))
    
    bill_names_in_text = []
    for text in substituted_original:
        bills = find_bill_names(text)
        bill_names_in_text.append(bills)

    combined["bill_names"] = bill_names_in_text
    
    #return enhance_dictionary_helper(combined)
    return collapse_dictionary(enhance_dictionary_helper(combined))
    
# given a list of dictionaries, return a 
def predict(new_transcript):
    processed_new_transcript = process_raw_data(new_transcript)

    transition_dictionary = {}

    if (model_type == "NB"):
        transition_dictionary = predict_from_naive_bayes(processed_new_transcript)

    elif (model_type == "NN"):
        raise Exception("Neural network not supported yet.")
        predict_from_neural_network()

    else:
        raise Exception("Model type not defined.")

    enhanced_dictionary = enhance_dictionary(new_transcript, transition_dictionary)
    return enhanced_dictionary
    

def main():
    new_transcript = pd.read_csv('../data/cleaned/transcript_witheld.csv', sep="~") \
        .to_dict('records')
    transition_dictionary = predict(new_transcript)
    enhanced_dictionary = enhance_dictionary(new_transcript, transition_dictionary)
    
    #print(enhanced_dictionary)
    i = 0
    for v in enhanced_dictionary:
        if v["suggested_bill"] != "NONE":
            i += 1
            print(v)
    
    print("number that have a suggested bill: ")
    print(i)

    #test = 'I urge a strong aye vote on assembly bill number 329 ab-329.'
    #find_bill_names(test)
    
if __name__ == "__main__":
    main()
