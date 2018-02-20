# DigitalDemocracyCapstone

## raw_and_upleveled_data_cleaning.py

Takes original raw transcripts and upleveled transcripts from data/original/ and generates files in data/cleaned/ in a more usable format.  The generated files are:
- data/cleaned/raw.csv - the basic utterances and times from the raw transcript
- data/cleaned/bill_start_end_times.csv - each row contains a bill id, video id, and the times in which the discussion of the bill started and stopped

## create_training_set_with_transitions.py

Generates the training sets by appending a column called 'transition_value' to training_data/cleaned/raw.csv.  The following training sets are generated:

#### training_utterances_binary.csv

Utterances are marked with a 1 if they are a transition and a 0 if they are not.

#### training_utterances_tertiary.csv

Utterances are marked with a 1 if they are the start of a bill, 2 if they are the end of a bill, and 0 otherwise.

#### training_utterances_n_range.csv

Utterances are marked with a 1 if they are within an n utterance range of a transition, and 0 otherwise.

#### training_utterances_n_range_collapsed.csv

All utterances within a "transition range" (within n utterances of a transition) are collapsed into a single utterance, marked with 1.  All other utterances are marked with 0.