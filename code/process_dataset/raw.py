import datetime
import math
import numpy as np
import pandas as pd
import sys
import os

from bs4 import BeautifulSoup

from code.process_dataset.bill_id_replacement import replace_bill_ids

# parse a string 00:00:00.470 to hours, minutes, seconds
# return time in seconds
def parse_time(time):
    time = time.split(":")
    hours = int(time[0])
    minutes = int(time[1])
    seconds = int(float(time[2]))
    
    return (hours*360)+(minutes*60)+seconds


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


# adds the prefix POST to all utterances n after
# adds the prefix PRE to all utterances n before
# a transition phrase
def add_context_naive_bayes(n, filename):
    n_range = pd.read_csv(filename, sep="~")
    
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
                    text += ' '.join(["POST-" + element for element in transition_text[i+x].split()])
                if (x < 0):
                    text += ' '.join(["PRE-" + element for element in transition_text[i+x].split()])
                else:
                    text += " {0} ".format(str(transition_text[i+x]))
                    
        new_transition_text.append(text)
    
    n_range.drop(['text'], axis=1, inplace=True)
    n_range['text'] = new_transition_text
    
    n_range.to_csv(filename, sep="~", index=False)


# creates three columns:
# utterance text
# pre-utterance text
# post-utterance text
def add_context_neural_network(n, input_file):
    n_range = pd.read_csv(cleaned_raw_bill_id_replaced_filename, sep="~")
    
    transition_text = n_range['text']
    new_transition_text = []
    pre_transition_text = []
    post_transition_text = []

    length = len(n_range)
    
    for i in range(length):
        # get the phrases in the window
        target_text = ''
        pre = ''
        post = ''
        
        for x in range(-n, n+1):
            # window is within range of the dataframe
            if (i + x >= 0 and i + x < length):
                if (x > 0):
                    post += ' '.join(["POST-" + element for element in transition_text[i+x].split()])
                if (x < 0):
                    pre += ' '.join(["PRE-" + element for element in transition_text[i+x].split()])
                else:
                    target_text += " {0} ".format(str(transition_text[i+x]))
                    
        new_transition_text.append(target_text)
        pre_transition_text.append(pre)
        post_transition_text.append(post)
    
    n_range.drop(['text'], axis=1, inplace=True)
    n_range['text'] = new_transition_text
    n_range['post_text'] = post_transition_text
    n_range['pre_text'] = pre_transition_text
    
    #n_range.to_csv(output_file, sep="~", index=False)
    return n_range
    
    
def raw_data_processing(input_filename, output_filename="../../data/cleaned/raw_cleaned.csv",
                        withheld_filename="../../data/cleaned/raw_cleaned_withheld.csv", withheld_ids=[],
                        input_sep="~~~~~", model_type="NB"):
    
    TEMP = "temp_raw_cleaned.csv"
    
    print("Parsing raw transcript...")
    raw = pd.read_table(input_filename, sep=input_sep, engine="python")
    cleaned_raw = parse_raw_data(raw)
    cleaned_raw = cleaned_raw.sort_values(["video_id", "start"])
    
    cleaned_raw_nonwithheld = cleaned_raw[~cleaned_raw['video_id'].isin(withheld_ids)]
    cleaned_raw_nonwithheld.to_csv(TEMP, sep="~", index=False)
    if (len(withheld_ids) > 0):
        cleaned_raw_withheld = cleaned_raw[cleaned_raw['video_id'].isin(withheld_ids)]
        cleaned_raw_withheld[["start", "end", "video_id", "text"]].to_csv(withheld_filename, sep='~', index=False)
        
    print("...Raw transcript parsed, beginning text formatting and bill replacement...")
    with open(TEMP, 'r') as old:
        with open(output_filename, 'w') as new:
            # consume/write headings
            h = old.readline()
            new.write(h)
            
            replace_bill_ids(old, new)
            
    print("...Text formatted and bills replaced, adding context...")
    add_context_naive_bayes(5, output_filename)

    print("...Context added... Raw transcript processing complete.\n")
    os.remove(TEMP)

# raw_data_processing("../../data/original/raw.txt",  "../../data/cleaned/raw_cleaned.csv")