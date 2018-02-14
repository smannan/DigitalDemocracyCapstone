# DigitalDemocracyCapstone

## raw_and_upleveled_data_cleaning.py

Takes original raw transcripts and upleveled transcripts from data/original/ and generates files in data/cleaned/ in a more usable format.  The generated files are:
- data/cleaned/raw.csv - the basic utterances and times from the raw transcript
- data/cleaned/bill_start_end_times.csv - each row contains a bill id, video id, and the times in which the discussion of the bill started and stopped

## create_training_set_with_transitions.py

Generates the training set by appending a 0, 1, or 2 to data/cleaned/raw.csv.  A 0 means the utterance is not a transition to a new bill, a 1 means the utterance starts a bill discussion, and a 2 means the utterance ends a bill discussion.  This training set is output to data/training/training_utterances_binary.csv.