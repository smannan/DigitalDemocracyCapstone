import sys
import pandas as pd


def get_transition_value(line, video_id, bill_start_time, bill_end_time):
    utterance_start_time = int(line.split("~")[0])
    utterance_end_time = int(line.split("~")[1])
    utterance_video_id = line.split("~")[2]
    
    if (video_id == utterance_video_id and
        utterance_end_time > bill_start_time and
        utterance_start_time < bill_end_time):
        return 1
    else:
        return 0
    
    
def separate_pre_post_cols(line):
    line = line[0].split(" ")
    
    pres = ' '.join([word for word in line if "PRE-" in word]).strip()
    posts = ' '.join([word for word in line if "POST-" in word]).strip()
    text = ' '.join([word for word in line if ("POST-" not in word) and ("PRE-" not in word)]).strip()
    
    return (pres, posts, text)

def write_pre_post_cols(line, transition_value):
    line = line.split("~")
    pres, posts, text = separate_pre_post_cols(line[3:])
    final = ["~".join(line[:3]), pres, text, posts, str(transition_value)]
    
    return "~".join(final) + "\n"


def mark_transition_lines(raw, bill_times, out, separate_pre_post=False):
    bill_times_splits = bill_times.readline().split("~")
    for line in raw:
        transition_value = get_transition_value(line, bill_times_splits[2], int(bill_times_splits[3]), int(bill_times_splits[4]))
        if (transition_value == 1):
            bill_line = bill_times.readline()
            if (bill_line == ""):
                bill_times_splits = [-1, -1, -1, -1, -1, -1]
            else:
                bill_times_splits = bill_line.split("~")
                
        if (separate_pre_post):
            final = write_pre_post_cols(line, transition_value)
            out.write(final)
            
        else:
            out.write(line.rstrip('\n') + "~" + str(transition_value) + "\n")
            

def create_training_data(cleaned_raw_filename, bill_times_table_filename, output_filename, separate_pre_post=False):
    print("Creating training data...")
    
    with open(cleaned_raw_filename, 'r') as raw:
        with open(bill_times_table_filename, 'r') as bill_times:
            with open(output_filename, 'w') as out:
                #consume/write headings
                raw.readline()
                bill_times.readline()
                out.write("start~end~video_id~text~transition_value\n")

                #mark the transitions
                mark_transition_lines(raw, bill_times, out, separate_pre_post=separate_pre_post)   
    
    print("...Training data created.\n")