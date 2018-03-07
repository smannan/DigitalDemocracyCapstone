
# coding: utf-8

# In[ ]:

import datetime
import math
import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup


# In[ ]:

original_raw_filename = "UNDEFINED"
original_upleveled_filename = "UNDEFINED"
original_upleveled_sorted_filename = "UNDEFINED"
cleaned_raw_filename = "UNDEFINED"
cleaned_raw_bill_id_replaced_filename="UNDEFINED"
cleaned_upleveled_filename = "UNDEFINED"
bill_start_end_times_all_filename = "UNDEFINED"
bill_start_end_times_longest_filename = "UNDEFINED"

with open("CONSTANTS") as constants_file:
    for line in constants_file:
        line_splits = line.rstrip("\n").split("=")
        
        if (line_splits[0] == "ORIGINAL_RAW"):
            original_raw_filename = line_splits[1]
        elif (line_splits[0] == "ORIGINAL_UPLEVELED"):
            original_upleveled_filename = line_splits[1]
        elif (line_splits[0] == "ORIGINAL_UPLEVELED_SORTED"):
            original_upleveled_sorted_filename = line_splits[1]
        elif (line_splits[0] == "CLEANED_RAW"):
            cleaned_raw_filename = line_splits[1]
        elif (line_splits[0] == "CLEANED_RAW_BILL_ID_REPLACED"):
            cleaned_raw_bill_id_replaced_filename = line_splits[1]
        elif (line_splits[0] == "CLEANED_UPLEVELED"):
            cleaned_upleveled_filename = line_splits[1]
        elif (line_splits[0] == "BILL_START_END_TIMES_ALL"):
            bill_start_end_times_all_filename = line_splits[1]
        elif (line_splits[0] == "BILL_START_END_TIMES_LONGEST"):
            bill_start_end_times_longest_filename = line_splits[1]


# # Raw Processing

# In[ ]:

raw = pd.read_table(original_raw_filename, sep='~~~~~', engine='python')
raw.head()


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


# In[ ]:

cleaned_raw = parse_raw_data(raw)
cleaned_raw.sort_values(["video_id", "start"]).to_csv(cleaned_raw_filename, sep="~", index=False)
cleaned_raw.head()


# ## Text Formatting and Bill Id Replacement

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
        
    print("Length of Bill Patterns Replaced\n1: " + str(t1) + "\n2: " + str(t2) + "\n3: " + str(t3) + "\n4: " + str(t4))
    return transition_window_list


# In[ ]:

transition_window_list = [] #not currently used, but is available for use

with open(cleaned_raw_filename, 'r') as old:
    with open(cleaned_raw_bill_id_replaced_filename, 'w') as new:
        # consume/write headings
        h = old.readline()
        new.write(h)
            
        #actually iterate through the file
        transition_window_list = replace_bill_ids(old, new)


# # Upleveled Processing

# In[ ]:

upleveled = pd.read_table(original_upleveled_filename, sep='~~~~~', engine='python')
upleveled = upleveled.sort_values(["video_id", "hearing_id", "speaker_start_time"])
upleveled.to_csv(original_upleveled_sorted_filename, sep="~", index=False)


# In[ ]:

def tag_bill_change_lines(original, cleaned):
    line = original.readline()
    current_bill_id = line.split("~")[0]
    i = 0
    cleaned.write(line.rstrip("\n") + "~0\n")
    
    for line in original:
        line_splits = line.split("~")
        
        if (line_splits[0] != current_bill_id):
            current_bill_id = line_splits[0]
            i += 1
        
        cleaned.write(line.rstrip("\n") + "~" + str(i) + "\n")


# In[ ]:

with open(original_upleveled_sorted_filename, 'r') as original:
    with open(cleaned_upleveled_filename, 'w') as cleaned:
        #consume/write headings
        h = original.readline()
        cleaned.write(h.rstrip("\n") + "~bill_change_tag\n")
            
        tag_bill_change_lines(original, cleaned)


# In[ ]:

tagged_upleveled = pd.read_table(cleaned_upleveled_filename, sep='~')


# In[ ]:

bill_start_times = tagged_upleveled.groupby(["bill_change_tag"]).head(1)
bill_end_times = tagged_upleveled.groupby(["bill_change_tag"]).tail(1)
bill_start_end_times = pd.merge(bill_start_times[["bill_id", "hearing_id", "video_id", "speaker_start_time", "bill_change_tag"]],
                                bill_end_times[["speaker_end_time", "bill_change_tag"]],
                                on=["bill_change_tag"]).drop(["bill_change_tag"], axis=1)
bill_start_end_times["length"] = bill_start_end_times["speaker_end_time"] - bill_start_end_times["speaker_start_time"]
bill_start_end_times = bill_start_end_times.sort_values(["video_id", "speaker_start_time"])


# In[ ]:

longest_bill_discussions = bill_start_end_times.sort_values(["bill_id", "length"]).groupby(["bill_id"]).tail(1)
longest_bill_discussions = longest_bill_discussions.sort_values(["video_id", "speaker_start_time"])


# In[ ]:

bill_start_end_times.to_csv(bill_start_end_times_all_filename, sep="~", index=False)
longest_bill_discussions.to_csv(bill_start_end_times_longest_filename, sep="~", index=False)


# In[ ]:



