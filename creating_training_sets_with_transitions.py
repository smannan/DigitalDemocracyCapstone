
# coding: utf-8

# In[ ]:

import pandas as pd


# In[ ]:

cleaned_raw_filename = "data/cleaned/raw.csv"
bill_start_end_times_filename = "data/cleaned/bill_start_end_times.csv"

training_output_tertiary_filename = "data/training/training_utterances_tertiary.csv"
training_output_binary_filename = "data/training/training_utterance_binary.csv"
training_output_n_range_filename = "data/training/training_utterances_n_range.csv"
training_output_n_range_collapsed_filename = "data/training/training_utterances_n_range_collapsed.csv"


# # Mark Transition Lines (Tertiary)

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
        if (transition_value == 2):
            bill_times_splits = bill_times.readline().split("~")
            
        print(line.rstrip('\n') + "~" + str(transition_value), file=out)


# In[ ]:

with open(cleaned_raw_filename, 'r') as raw:
    with open(bill_start_end_times_filename, 'r') as bill_times:
        with open(training_output_tertiary_filename, 'w') as out:
            #consume/write headings
            raw.readline()
            bill_times.readline()
            out.write("start~end~text~video_id~transition_value\n")
            
            #mark the transitions
            mark_transition_lines(raw, bill_times, out)


# # Mark Transition Lines (Binary)

# In[ ]:

binary = pd.read_csv(training_output_tertiary_filename, sep="~")
binary[binary["transition_value"] != 0]
binary.to_csv(training_output_binary_filename, sep="~", index=False)


# # Mark Transition Lines (N Range)

# In[ ]:

n = 5


# In[ ]:

n_range = pd.read_csv(training_output_binary_filename, sep="~")


# In[ ]:

transition_indexes = n_range.index[n_range["transition_value"] == 1].tolist()
new_transition_indexes = []

length = len(transition_indexes)
for i in transition_indexes:
    for x in range(-n, n):
        if (i + x >= 0 and i + x < length):
            new_transition_indexes.append(i + x)


# In[ ]:

n_range.loc[new_transition_indexes, "transition_value"] = 1


# In[ ]:

n_range.to_csv(training_output_n_range_filename, sep="~", index=False)


# # Collapse Transition Lines (N Range)

# In[ ]:

def collapse_transitions(uncollapsed, collapsed):
    accumulated_text = ""
    accumulating = False
    
    for line in uncollapsed:
        split_line = line.split("~")
        transition_value = int(split_line[4])
        text = split_line[2] + " "
        
        if transition_value == 1 and accumulating:
            accumulated_text = accumulated_text + text
        elif transition_value == 1 and not accumulating:
            accumulating = True
            accumulated_text = accumulated_text + text
        elif transition_value == 0 and accumulating:
            collapsed.write(split_line[0] + "~" + split_line[1] + "~" +
                            accumulated_text + "~" + split_line[3] + "~1\n")
            collapsed.write(line)
            accumulating = False
            accumulated_text = ""
        else:
            collapsed.write(line)


# In[ ]:

with open(training_output_n_range_filename, 'r') as uncollapsed:
    with open(training_output_n_range_collapsed_filename, 'w') as collapsed:
        #consume/write headings
        h = uncollapsed.readline()
        collapsed.write(h)
            
        #collapse transitions
        collapse_transitions(uncollapsed, collapsed)

