
# coding: utf-8

# In[ ]:

import datetime
import math
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup


# In[ ]:

original_raw_filename = "data/original/raw.txt"
original_upleveled_filename = "data/original/upleveled.txt"
cleaned_raw_filename = "data/cleaned/raw.csv"
bill_start_end_times_filename = "data/cleaned/bill_start_end_times.csv"


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
upleveled.head()


# In[ ]:

bill_start_times = upleveled.sort_values(["video_id", "bill_id", "speaker_start_time"]).groupby(["bill_id", "hearing_id", "video_id"]).head(1)
bill_end_times = upleveled.sort_values(["video_id", "bill_id", "speaker_start_time"]) .groupby(["bill_id", "hearing_id", "video_id"]).tail(1)
bill_start_end_times = pd.merge(bill_start_times[["bill_id", "video_id", "speaker_start_time"]],
                                bill_end_times[["bill_id", "video_id", "speaker_end_time"]],
                                on=["bill_id", "video_id"])


# In[ ]:

bill_start_end_times.sort_values(["video_id", "speaker_start_time"]).to_csv(bill_start_end_times_filename, sep="~", index=False)
bill_start_end_times.head()

