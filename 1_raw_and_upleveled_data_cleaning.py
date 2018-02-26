
# coding: utf-8

# In[ ]:

import datetime
import math
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup


# In[ ]:

original_raw_filename = "UNDEFINED"
original_upleveled_filename = "UNDEFINED"
original_upleveled_sorted_filename = "UNDEFINED"
cleaned_raw_filename = "UNDEFINED"
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
