
# coding: utf-8

# In[1]:

import datetime
import math
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup


# # Raw Processing

# In[2]:

raw = pd.read_table("digitaldemocracy_ds_capstone_2018/dd_capstone_raw_transcripts.txt", sep='~~~~~', engine='python')
raw.head()


# ### Tidy the raw data

# ### Beautiful Soup to parse raw_transcript for each video_id

# In[65]:

# parse a string 00:00:00.470 to hours, minutes, seconds, ms
# return a new datetime
def parse_time(time):
    time = time.split(":")
    hours = int(time[0]) # hours
    minutes = int(time[1]) # minutes
    
    seconds = int(float(time[2])) # 00.470 or 2.470
    ms = int(math.fabs(seconds - float(time[2])) * 100)
    
    return datetime.time(hours, minutes, seconds, ms)


# In[79]:

def parse_raw_data(raw):
    r = raw['raw_transcript']
    ids = raw['video_id']
    res = {'start':[], 'end':[], 'text':[], 'vid': []}
    
    for transcript, vid in zip(r, ids):
        #print(vid)
        soup = BeautifulSoup(transcript, "lxml")
        letters = soup.find_all("p")

        for p in letters[1:]:
            res['start'].append(parse_time(p.get('begin')))
            res['end'].append(parse_time(p.get('end')))
            res['text'].append(p.contents[0])
            res['vid'].append(vid)

    tidy = pd.DataFrame(res, columns=['start', 'end', 'text', 'vid'])
    return (tidy)


# In[80]:

tidy = parse_raw_data(raw)
tidy.head()


# In[81]:

tidy.to_csv('raw_transcript_tidy.csv')


# # Upleveled Processing

# In[3]:

upleveled = pd.read_csv("digitaldemocracy_ds_capstone_2018/dd_capstone_data.txt", sep='~~~~~', engine='python')
upleveled.head()


# In[7]:

len(upleveled)


# ### Mark transistions between bills

# In[8]:

bids = upleveled['bill_id']
transitions = [0] * len(bids)
transitions[0] = 1

i = 1
while i < len(bids) - 1:
    if bids[i] != bids[i+1]:
        transitions[i] = 1
        transitions[i+1] = 0
        i += 2
        
    else:
        transitions[i] = 0
        i += 1


# In[9]:

upleveled['transition'] = transitions


# In[10]:

upleveled[upleveled['transition'] == 1]


# In[113]:

upleveled.to_csv('dd_capstone_data_transitions.csv')


# ### TODO for Monday: Mark text in raw dataframe as transitions or not based on text from upleveled dataframe
# ### Split raw df into training + testing with balanced class labels of transitions
# ### Vectorize text for Naive Bayes classifier

# # Training/Test Set Assembly

# In[9]:

#merge raw/upleveled on bill/hearing/video
#aka append timestamps from upleveled to raw


# In[87]:

#bill_start_times = upleveled.sort_values(["video_id", "hearing_id", "bill_id", "speaker_start_time"]).groupby(["bill_id", "hearing_id", "video_id"]).head(1)
#bill_end_times = upleveled.sort_values(["video_id", "hearing_id", "bill_id", "speaker_start_time"]) .groupby(["bill_id", "hearing_id", "video_id"]).tail(1)
#bill_times = pd.merge(bill_start_times[["bill_id", "hearing_id", "video_id", "speaker_start_time"]],
#                      bill_end_times[["bill_id", "speaker_end_time"]], on="bill_id")


# In[88]:

# bill_times.sort_values("speaker_start_time").head() #these are the timestamps of the bill discussion for a unique bill_id, hearing_id, and video_id


# In[ ]:



