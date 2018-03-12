
# coding: utf-8

# # Create Training Data

# In[ ]:

import sys
import pandas as pd


# ### Determine inputs.

# In[ ]:

default_cleaned_raw_filename = "../data/cleaned/cleaned_raw.csv"
default_temp_folder = "."
default_output_folder = "."


# In[ ]:

cleaned_raw_filename = default_cleaned_raw_filename
temp_folder = default_temp_folder
output_folder = default_output_folder

if sys.argv[0] != '/opt/conda/lib/python3.5/site-packages/ipykernel_launcher.py':
    if len(sys.argv) > 1:
        cleaned_raw_filename = sys.argv[1]
    if len(sys.argv) > 2:
        temp_folder = sys.argv[2]
    if len(sys.argv) > 3:
        output_folder = sys.argv[3]


# In[ ]:

#constants
TEMP_BILL_START_END_TIMES_ALL_FILENAME = temp_folder + "/temp_bill_start_end_times_all.csv"
TEMP_BILL_START_END_TIMES_LONGEST_FILENAME = temp_folder + "/temp_bill_start_end_times_longest.csv"

OUTPUT_TRAINING_BINARY_FILENAME = output_folder + "/training_output_binary_filename.csv"
OUTPUT_TRAINING_N_RANGE_FILENAME = output_folder + "/training_output_n_range_filename.csv"
OUTPUT_TRAINING_N_RANGE_COLLAPSED_FILENAME = output_folder + "/training_output_n_range_collapsed_filename.csv"


# ### Determine which bill times table will be used (recommended: longest).

# In[ ]:

bill_start_end_times_filename = TEMP_BILL_START_END_TIMES_LONGEST_FILENAME


# ### Mark transition lines (binary).

# In[ ]:

print("...Creating training data...")


# In[ ]:

def get_transition_value(line, bill_start_time, bill_end_time):
    utterance_start_time = int(line.split("~")[0])
    utterance_end_time = int(line.split("~")[1])
    
    if (utterance_end_time > bill_start_time and bill_end_time > utterance_start_time):
        return 1
    else:
        return 0


# In[ ]:

def mark_transition_lines(raw, bill_times, out):
    bill_times_splits = bill_times.readline().split("~")
    for line in raw:
        transition_value = get_transition_value(line, int(bill_times_splits[3]), int(bill_times_splits[4]))
        if (transition_value == 1):
            bill_line = bill_times.readline()
            if (bill_line == ""):
                bill_times_splits = [-1, -1, -1, -1, -1, -1]
            else:
                bill_times_splits = bill_line.split("~")
            
        out.write(line.rstrip('\n') + "~" + str(transition_value) + "\n")


# In[ ]:

with open(cleaned_raw_filename, 'r') as raw:
    with open(bill_start_end_times_filename, 'r') as bill_times:
        with open(OUTPUT_TRAINING_BINARY_FILENAME, 'w') as out:
            #consume/write headings
            raw.readline()
            bill_times.readline()
            out.write("start~end~text~video_id~transition_value\n")
            
            #mark the transitions
            mark_transition_lines(raw, bill_times, out)


# ### Mark transition lines (n-range).

# In[ ]:

n = 5


# In[ ]:

n_range = pd.read_csv(OUTPUT_TRAINING_BINARY_FILENAME, sep="~")


# In[ ]:

transition_indexes = n_range.index[n_range["transition_value"] == 1].tolist()
new_transition_indexes = []

length = len(n_range)
for i in transition_indexes:
    for x in range(-n, n+1):
        if (i + x >= 0 and i + x < length):
            new_transition_indexes.append(i + x)
            
n_range.loc[new_transition_indexes, "transition_value"] = 1

n_range.to_csv(OUTPUT_TRAINING_N_RANGE_FILENAME, sep="~", index=False)


# ### Collapse transition lines (n-range).

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

with open(OUTPUT_TRAINING_N_RANGE_FILENAME, 'r') as uncollapsed:
    with open(OUTPUT_TRAINING_N_RANGE_COLLAPSED_FILENAME, 'w') as collapsed:
        #consume/write headings
        h = uncollapsed.readline()
        collapsed.write(h)
            
        #collapse transitions
        collapse_transitions(uncollapsed, collapsed)


# In[ ]:

print("Training data complete.")


# ### Verification

# In[ ]:

binary = pd.read_csv(OUTPUT_TRAINING_BINARY_FILENAME, sep="~")
bill_times = pd.read_csv(bill_start_end_times_filename, sep="~")


# In[ ]:

assert(len(binary[binary["transition_value"]==1])==len(bill_times))

