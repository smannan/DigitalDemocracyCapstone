
# coding: utf-8

# In[ ]:

import pandas as pd


# In[ ]:

cleaned_raw_filename = "data/cleaned/raw.csv"
bill_start_end_times_filename = "data/cleaned/bill_start_end_times.csv"
training_output_filename = "data/training/training_utterances_binary.csv"


# # Mark Transition Lines

# In[ ]:

def get_transition_value(line, start_time, end_time):
    if (start_time >= int(line.split("~")[0]) and start_time < int(line.split("~")[1])):
        return 1
    elif (end_time >= int(line.split("~")[0]) and end_time < int(line.split("~")[1])):
        return 2
    else:
        return 0


# In[ ]:

def mark_transition_lines(raw, bill_times, out):
    bill_times_splits = bill_times.readline().split("~")
    for line in raw:
        transition_value = get_transition_value(line, int(bill_times_splits[2]), int(bill_times_splits[3]))
        if (transition_value != 0):
            bill_times_splits = bill_times.readline().split("~")
            
        print(line.rstrip('\n') + "~" + str(transition_value), file=out)


# In[ ]:

with open(cleaned_raw_filename, 'r') as raw:
    with open(bill_start_end_times_filename, 'r') as bill_times:
        with open(training_output_filename, 'w') as out:
            #consume/write headings
            raw.readline()
            bill_times.readline()
            out.write("start~end~text~video_id~transition_value\n")
            #mark the transitions
            mark_transition_lines(raw, bill_times, out)

