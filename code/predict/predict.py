import math
import re
import sys

sys.path.insert(0, '../..')
from code.process_dataset.raw import *
from code.process_dataset.bill_id_replacement import replace_bill_ids
from predict_transitions import *

# new_transcript is a list of dictionaries
def process_raw_transcript_data(new_transcript):
    ### Determine the raw .srt file that will be processed.

    cleaned_raw_filename = "cleaned_transcript.csv"
    cleaned_raw_bill_id_replaced_filename = cleaned_raw_filename + "2"

    cleaned_raw = pd.DataFrame(new_transcript)
    cleaned_raw.sort_values(["video_id", "start"]) \
     .to_csv(cleaned_raw_filename, sep="~", index=False)

    with open(cleaned_raw_filename, 'r') as old:
        with open(cleaned_raw_bill_id_replaced_filename, 'w') as new:
            # consume/write headings
            h = old.readline()
            new.write(h)
            
            replace_bill_ids(old, new)

    add_context_naive_bayes(5, cleaned_raw_bill_id_replaced_filename)

    processed_new_transcript = pd.read_csv(cleaned_raw_bill_id_replaced_filename, sep="~")
    return processed_new_transcript


def predict(new_video_transcript_dictionary, model_type,
            model_folder="../../model"):
    
    processed_new_transcript = process_raw_transcript_data(new_video_transcript_dictionary)
    
    return predict_transitions(new_video_transcript_dictionary, processed_new_transcript, model_type, model_folder)


def print_withheld_accuracy(transition_dictionary, bill_table, proximity_time):
    r = 0
    bill_ids = list(bill_table["bill_id"])
    times = list(bill_table["speaker_start_time"])
    
    #p = re.compile("[A-Z]+[0-9]+")
    #p.search(s)
    
    for i in range(len(times)):
        for entry in transition_dictionary:
            if (math.fabs(entry["start"] - times[i]) < proximity_time):
                r += 1
                break
    
    print("Recall within {0} seconds: {1}/{2} = {3}".format(proximity_time, r, len(times), 1.0*r/len(times)))
    print("Precision within {0} seconds: {1}/{2} = {3}".format(proximity_time, r, len(transition_dictionary),
                                                               1.0*r/len(transition_dictionary)))
    print()
    

def evaluate_withheld_transcripts(withheld_training_transcripts, model_type,
                                  withheld_bill_times_table="../../data/cleaned/upleveled_bill_times_table_withheld.csv",
                                  model_folder="../../model"):
    new_transcript = pd.read_csv(withheld_training_transcripts, sep="~")[["start", "end", "text", "video_id"]]
    bill_times_table = pd.read_csv(withheld_bill_times_table, sep="~")
    video_ids = np.unique(new_transcript[["video_id"]])
    
    for video_id in video_ids:
        new_transcript_subset = new_transcript[new_transcript["video_id"]==video_id].to_dict("records")
        bill_times_table_subset = bill_times_table[bill_times_table["video_id"]==video_id]
        
        print("\n--------------------")
        print("Video ID: " + str(video_id))
        print("--------------------\n")
        
        transition_dictionary = predict(new_transcript_subset, model_type)
        enhanced_dictionary = enhance_dictionary(new_transcript_subset, transition_dictionary)
        shortened_dictionary = remove_unknown_suggested_bills(enhanced_dictionary)
        #shortened_dictionary = enhanced_dictionary
        
        print("Actual Bill Discussion Starts:\n")
        
        bill_ids = list(bill_times_table_subset["bill_id"])
        times = list(bill_times_table_subset["speaker_start_time"])
        for i in range(len(times)):
            print("{0} - {1}".format(bill_ids[i], times[i]))
        
        print()
        print("Predicted Transitions:\n")
        for entry in shortened_dictionary:
            print(entry)
        
        print()
        print_withheld_accuracy(shortened_dictionary, bill_times_table_subset, 15)
        print_withheld_accuracy(shortened_dictionary, bill_times_table_subset, 30)
        print_withheld_accuracy(shortened_dictionary, bill_times_table_subset, 60)
