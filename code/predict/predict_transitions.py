import datetime
import math
import numpy as np
import pandas as pd
import os
import pickle
import re
import sys
import textedit
from bs4 import BeautifulSoup

from code.process_dataset.bill_id_replacement import find_bill_names


THRESHOLD_PROBABILITY = .5


def predict_entire_transcript(transcripts, model, count_vect):
    transcripts_test = count_vect.transform(transcripts['text'])
    
    probs = model.predict_proba(transcripts_test)
    preds = [1 if p[1] > THRESHOLD_PROBABILITY else 0 for p in probs]
    
    assert len(preds) == transcripts_test.shape[0]
    
    return preds


def predict_from_naive_bayes(transcript, model_folder):
    model = pickle.load(open(model_folder + "/nb_model.p", "rb"))
    count_vect = pickle.load(open(model_folder + "/nb_count_vect.p", "rb"))

    prediction_values = predict_entire_transcript(transcript, model, count_vect)
    transcript["prediction"] = prediction_values
    predicted_transitions = transcript[transcript["prediction"]==1]
    transition_dictionary = predicted_transitions[["video_id", "start", "end", "text"]].to_dict(orient="records")
    
    return transition_dictionary


def predict_from_neural_network():
    pass


def remove_unknown_suggested_bills(old_dictionary):
    new_dictionary = []
        
    for t in old_dictionary:
        if len(t["suggested_bill"]) > 0:
            new_dictionary.append(t)
            
    return new_dictionary


# enhanced dictionary = list of dicts containing transition utterances and
# their associated bill names
# collapse neighboring transitions by taking the first transition in a
# sequence of adjacent transitions
# captures all suggested bill names in a sequence
# returns a list of dictionaries
def collapse_dictionary(enhanced_dictionary):
    res = []
    i = 0
    j = 0
    n = len(enhanced_dictionary)
    epsilon = 5

    while i < n:
        all_bill_names = []
        j = i + 1
        row = enhanced_dictionary[i]

        all_bill_names += (enhanced_dictionary[i]['suggested_bill'])

        while j < n and math.fabs(enhanced_dictionary[j]['start']
         - enhanced_dictionary[i]['end']) < epsilon:
            j += 1
            i += 1
            all_bill_names += (enhanced_dictionary[i]['suggested_bill'])

        i = j
        row['suggested_bill'] = all_bill_names
        res.append(row)

    return res


def enhance_dictionary_helper(combined): 
    enhanced_dictionary = []
    
    start_list = list(combined["start"])
    end_list = list(combined["end"])
    video_id_list = list(combined["video_id"])
    original_text_list = list(combined["text_original"])
    modified_text_list = list(combined["text_modified"])
    bill_names_list = list(combined["bill_names"])
    
    found_associated_bill_id = True
    i = 0
    while (i < len(original_text_list)):
        # if the utterance is not a transition utterance, continue
        if (pd.isnull(modified_text_list[i])):
            i += 1
            continue
        
        # otherwise, if there is a bill id immediately associated with the utterance,
        # input it into the dictionary, then continue
        
        if (len(bill_names_list[i]) > 0):
            enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':bill_names_list[i]})
            i += 1
            continue

        #otherwise, scan all table rows until the next transition utterance looking for
        #an associated bill id
        j = i+1
        while(j < len(original_text_list) and pd.isnull(modified_text_list[j])):
            if (len(bill_names_list[j]) > 0):
                enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':bill_names_list[j]})
                i = j+1
                found_associated_bill_id = False
                break
            j += 1
            
        if (not found_associated_bill_id):
            found_associated_bill_id = True
            continue
        
        #at this point, the transition utterance has no associated bill ids in the window
        #so we likely want to disregard the transition as a possibility
        #for now we simply enter it with the suggested_bill field having value []
        enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':[]})
        i = j
            
    return enhanced_dictionary

def enhance_dictionary(original_transcript, transition_dictionary):
    #transition_dictitionary format: [{video_id, start, end, text}]
    #original transcript format new_transcript = [{'start':'1', 'end':'9', 
    #'text':'we are starting', 'video_id':'1'}]
    
    original = pd.DataFrame.from_dict(original_transcript)
    transitions = pd.DataFrame.from_dict(transition_dictionary)
    
    combined = pd.merge(original, transitions, how='left', on=["video_id", "start", "end"],
                        suffixes=("_original", "_modified")) 
    substituted_original = textedit.correct_text(list(combined["text_original"]))
    
    bill_names_in_text = []
    for text in substituted_original:
        bills = find_bill_names(text)
        bill_names_in_text.append(bills)

    combined["bill_names"] = bill_names_in_text
    
    return collapse_dictionary(enhance_dictionary_helper(combined))
    
    
# given a list of dictionaries, return a dictionary of suspected transitions
def predict_transitions(original_new_transcript, processed_new_transcript, model_type, model_folder):

    transition_dictionary = {}

    if (model_type == "NB"):
        transition_dictionary = predict_from_naive_bayes(processed_new_transcript, model_folder)

    elif (model_type == "NN"):
        raise Exception("Neural network not supported yet.")
        predict_from_neural_network()

    else:
        raise Exception("Model type not defined.")

    enhanced_dictionary = enhance_dictionary(original_new_transcript, transition_dictionary)
    return enhanced_dictionary
    