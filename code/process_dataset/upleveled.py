import pandas as pd
import os

# tag lines with bill_change_tag, which increments every time the bill is changed.
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

def upleveled_data_processing(input_filename,
                              output_filename="../../data/cleaned/upleveled_bill_times_table.csv",
                              withheld_filename="../../data/cleaned/upleveled_bill_times_table_withheld.csv", withheld_ids=[],
                              input_sep="~~~~~"):
    TEMP1 = "temp_original_upleveled_sorted.csv"
    TEMP2 = "temp_marked_upleveled.csv"
    
    print("Parsing upleveled transcript...")
    upleveled = pd.read_table(input_filename, sep=input_sep, engine="python")
    upleveled = upleveled.sort_values(["video_id", "hearing_id", "speaker_start_time"])
    upleveled.to_csv(TEMP1, sep="~", index=False)
    
    print("...Creating bill times table...")
    with open(TEMP1, 'r') as original:
        with open(TEMP2, 'w') as cleaned:
            #consume/write headings
            h = original.readline()
            cleaned.write(h.rstrip("\n") + "~bill_change_tag\n")

            tag_bill_change_lines(original, cleaned)
            
    
    tagged_upleveled = pd.read_table(TEMP2, sep='~')
    
    bill_start_times = tagged_upleveled.groupby(["bill_change_tag"]).head(1)
    bill_end_times = tagged_upleveled.groupby(["bill_change_tag"]).tail(1)
    bill_start_end_times = pd.merge(bill_start_times[["bill_id", "hearing_id", "video_id", "speaker_start_time", "bill_change_tag"]],
                                    bill_end_times[["speaker_end_time", "bill_change_tag"]],
                                    on=["bill_change_tag"]).drop(["bill_change_tag"], axis=1)
    bill_start_end_times["length"] = bill_start_end_times["speaker_end_time"] - bill_start_end_times["speaker_start_time"]
    bill_start_end_times = bill_start_end_times.sort_values(["video_id", "speaker_start_time"])
    
    longest_bill_discussions = bill_start_end_times.sort_values(["bill_id", "length"]).groupby(["bill_id"]).tail(1)
    longest_bill_discussions = longest_bill_discussions.sort_values(["video_id", "speaker_start_time"])
    
    longest_bill_discussions_nonwithheld = longest_bill_discussions[~longest_bill_discussions["video_id"].isin(withheld_ids)]
    longest_bill_discussions_nonwithheld.to_csv(output_filename, sep="~", index=False)
    if (len(withheld_ids) > 0):
        longest_bill_discussions_withheld = longest_bill_discussions[longest_bill_discussions["video_id"].isin(withheld_ids)]
        longest_bill_discussions_withheld.to_csv(withheld_filename, sep='~', index=False)   
        
    print("...Upleveled data processing complete.\n")
    
    os.remove(TEMP1)
    os.remove(TEMP2)
    
    