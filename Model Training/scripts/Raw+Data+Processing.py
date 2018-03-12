
# coding: utf-8

# # Raw Data Processing

# In[ ]:

import datetime
import math
import numpy as np
import pandas as pd
import re
import sys

from bs4 import BeautifulSoup


# ### Determine the raw .srt file that will be processed.

# In[ ]:

default_original_raw_filename = "../data/original/raw.txt"
default_cleaned_raw_filename = "../data/cleaned/cleaned_raw.csv"


# In[ ]:

original_raw_filename = default_original_raw_filename
cleaned_raw_filename = default_cleaned_raw_filename

if sys.argv[0] != '/opt/conda/lib/python3.5/site-packages/ipykernel_launcher.py':
    if len(sys.argv) > 1:
        original_raw_filename = sys.argv[1]
    if len(sys.argv) > 2:
        cleaned_raw_filename = sys.argv[2]
        
cleaned_raw_bill_id_replaced_filename = cleaned_raw_filename + "2"


# ### Read in the raw transcript specified above, and verify it was read in correctly.

# In[ ]:

raw = pd.read_table(original_raw_filename, sep='~~~~~', engine='python')
raw.head()


# ### Define parsing functions.

# In[ ]:

# parse a string 00:00:00.470 to hours, minutes, seconds
# return time in seconds
def parse_time(time):
    time = time.split(":")
    hours = int(time[0])
    minutes = int(time[1])
    seconds = int(float(time[2])) 
    
    return (hours*360)+(minutes*60)+seconds


# In[ ]:

def parse_raw_data(raw):
    r = raw['raw_transcript']
    ids = raw['video_id']
    res = {'start':[], 'end':[], 'text':[], 'video_id': []}
    for transcript, vid in zip(r, ids):
        soup = BeautifulSoup(transcript, "lxml")
        letters = soup.find_all("p")

        for p in letters[1:]:
            res['start'].append(parse_time(p.get('begin')))
            res['end'].append(parse_time(p.get('end')))
            res['text'].append(p.contents[0])
            res['video_id'].append(vid)

    tidy = pd.DataFrame(res, columns=['start', 'end', 'text', 'video_id'])
    return (tidy)


# ### Define text formatting and bill replacement logic.  This converts all utterances to entirely lowercase, and replaces the following instances of words in an utterance with the tag BILL_ID.

# In[ ]:

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


# In[ ]:

def re_match_lists_helper(pattern_list, word_list):
    for p in range(len(pattern_list)):
        if not (re.match(pattern_list[p], word_list[p])):
            return False
    return True

def re_match_lists(pattern_list_list, word_list):
    for pl in range(len(pattern_list_list)):
        if (re_match_lists_helper(pattern_list_list[pl], word_list)):
            return True
    return False

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


# In[ ]:

def shift_words_over(words, word_ix, shift_amount):
    words_length = len(words)
    
    for i in range(word_ix, words_length - shift_amount):
        words[i] = words[i+shift_amount]
    while(len(words) > (words_length-shift_amount)):
        del words[-1]
        
    return words


# In[ ]:

def replace_bill_ids_in_utterance(utterance, last_bill_number, t1, t2, t3, t4):
    words = utterance.lower().split()
    utterance_length = len(words)
    word_ix = 0
    bill_id_replaced = False
    while(word_ix < utterance_length):
        if (word_ix < (utterance_length-3) and
            matches_any_4_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2],
                                         words[word_ix+3])):
            last_bill_number = words[word_ix+3]
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 3)
            utterance_length -= 3
            bill_id_replaced = True
            t4 += 1
        elif (word_ix < (utterance_length-2) and
              matches_any_3_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2])):
            last_bill_number = words[word_ix+2]
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 2)
            utterance_length -= 2
            bill_id_replaced = True
            t3 += 1
        elif (word_ix < (utterance_length-1) and
            matches_any_2_word_pattern(words[word_ix],
                                         words[word_ix+1])):
            last_bill_number = words[word_ix+1]
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 1)
            utterance_length -= 1
            bill_id_replaced = True
            t2 += 1
        elif (matches_any_1_word_pattern(words[word_ix])):
            last_bill_number = words[word_ix].split("[a-z]+")[-1]
            words[word_ix] = "<BILL_ID>"
            bill_id_replaced = True
            t1 += 1

        word_ix += 1
            
    return (" ".join(words), last_bill_number, bill_id_replaced, t1, t2, t3, t4)


# In[ ]:

def replace_bill_ids(old, new):
    t1 = 0  #keeps track of how many bill id replacements there were
    t2 = 0
    t3 = 0
    t4 = 0
    
    last_bill_number = ""
    last_bill_number_line = 0
    transition_window_list = []
    line_number = 0
    for line in old:
        line_splits = line.lower().rstrip("\n").split("~")
        
        (new_text, current_bill_number, bill_id_replaced, t1, t2, t3, t4) = replace_bill_ids_in_utterance(line_splits[2], last_bill_number, t1, t2, t3, t4)
        
        if (bill_id_replaced):
            if (current_bill_number != last_bill_number):
                transition_window_list.append((last_bill_number_line, line_number))
                last_bill_number = current_bill_number
                last_bill_number_line = line_number
            elif (current_bill_number == last_bill_number):
                last_bill_number_line = line_number
        
        new.write(line_splits[0] + "~" + line_splits[1] + "~" + new_text + "~" + line_splits[3] + "\n")
        line_number += 1
        
    #print("Length of Bill Patterns Replaced\n1: " + str(t1) + "\n2: " + str(t2) + "\n3: " + str(t3) + "\n4: " + str(t4))
    return transition_window_list


# ### Actually parse the raw transcript and replace bill ids.

# In[ ]:

print("...Parsing raw transcript...")

cleaned_raw = parse_raw_data(raw)
cleaned_raw.sort_values(["video_id", "start"]).to_csv(cleaned_raw_filename, sep="~", index=False)

print("...Raw transcript parsed, beginning text formatting and bill replacement...")

transition_window_list = [] #not currently used, but is available for use

with open(cleaned_raw_filename, 'r') as old:
    with open(cleaned_raw_bill_id_replaced_filename, 'w') as new:
        # consume/write headings
        h = old.readline()
        new.write(h)
            
        #actually iterate through the file
        transition_window_list = replace_bill_ids(old, new)

print("...Text formatting and bill replacement complete.")

cleaned_raw.head()


# In[ ]:



