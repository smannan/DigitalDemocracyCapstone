import re

# Define text formatting and bill replacement logic.  
# This converts all utterances to entirely lowercase, 
# and replaces the following instances of words in an 
# utterance with the tag BILL_ID.

bill_id_pattern_1_1 = "ab[0-9]+"
bill_id_pattern_1_2 = "sb[0-9]+"
bill_id_pattern_1_3 = "aca[0-9]+"
bill_id_pattern_1_4 = "acr[0-9]+"
bill_id_pattern_1_5 = "ajr[0-9]+"
bill_id_pattern_1_6 = "ar[0-9]+"
bill_id_pattern_1_7 = "hr[0-9]+"
bill_id_pattern_1_8 = "sca[0-9]+"
bill_id_pattern_1_9 = "scr[0-9]+"
bill_id_pattern_1_10 = "sjr[0-9]+"

bill_id_pattern_2_1 = ["ab", "[0-9]+"]
bill_id_pattern_2_2 = ["sb", "[0-9]+"]
bill_id_pattern_2_3 = ["aca", "[0-9]+"]
bill_id_pattern_2_4 = ["acr", "[0-9]+"]
bill_id_pattern_2_5 = ["ajr", "[0-9]+"]
bill_id_pattern_2_6 = ["ar", "[0-9]+"]
bill_id_pattern_2_7 = ["hr", "[0-9]+"]
bill_id_pattern_2_8 = ["sca", "[0-9]+"]
bill_id_pattern_2_9 = ["scr", "[0-9]+"]
bill_id_pattern_2_10 = ["sjr", "[0-9]+"]

bill_id_pattern_3_1 = ["assembly", "bill", "[0-9]+"]
bill_id_pattern_3_2 = ["senate", "bill", "[0-9]+"]

bill_id_pattern_4_1 = ["assembly", "bill", "number", "[0-9]+"]
bill_id_pattern_4_2 = ["senate", "bill", "number", "[0-9]+"]


def re_match_lists_helper(pattern_list, word_list):
    for p in range(len(pattern_list)):
        if not (re.match(pattern_list[p], word_list[p])):
            return False
    return True

def re_match_lists(pattern_list_list, word_list):
    # for each pattern
    for pl in range(len(pattern_list_list)):
        # match the specific pattern against the word
        if (re_match_lists_helper(pattern_list_list[pl], word_list)):
            return True
    return False

def re_find_lists(pattern_list_list, word):
    res = []

    for i in range(len(pattern_list_list)):
        #print (word, ' '.join(pattern_list_list[i]))

        p = re.compile(' '.join(pattern_list_list[i]))

        res += p.findall(word)

    if len(res) == 0:
        return ''

    return ' '.join(res)

def find_any_pattern(word, pattern_list):
    return re_find_lists(pattern_list, word)

def matches_any_4_word_pattern(word1, word2, word3, word4):
    pattern_list_list = [bill_id_pattern_4_1, bill_id_pattern_4_2]
    word_list = [word1, word2, word3, word4]
    
    return re_match_lists(pattern_list_list, word_list)

def matches_any_3_word_pattern(word1, word2, word3):
    pattern_list_list = [bill_id_pattern_3_1, bill_id_pattern_3_2]
    word_list = [word1, word2, word3]
    
    return re_match_lists(pattern_list_list, word_list)
    
def matches_any_2_word_pattern(word1, word2):
    pattern_list_list = [bill_id_pattern_2_1, bill_id_pattern_2_2,
                         bill_id_pattern_2_3, bill_id_pattern_2_4,
                         bill_id_pattern_2_5, bill_id_pattern_2_6,
                         bill_id_pattern_2_7, bill_id_pattern_2_8,
                         bill_id_pattern_2_9, bill_id_pattern_2_10]
    word_list = [word1, word2]
    
    return re_match_lists(pattern_list_list, word_list)

def matches_any_1_word_pattern(word):
    return (re.match(bill_id_pattern_1_1, word) or
            re.match(bill_id_pattern_1_2, word) or
            re.match(bill_id_pattern_1_3, word) or
            re.match(bill_id_pattern_1_4, word) or
            re.match(bill_id_pattern_1_5, word) or
            re.match(bill_id_pattern_1_6, word) or
            re.match(bill_id_pattern_1_7, word) or
            re.match(bill_id_pattern_1_8, word) or
            re.match(bill_id_pattern_1_9, word) or
            re.match(bill_id_pattern_1_10, word))

def shift_words_over(words, word_ix, shift_amount):
    words_length = len(words)
    
    for i in range(word_ix, words_length - shift_amount):
        words[i] = words[i+shift_amount]
    while(len(words) > (words_length-shift_amount)):
        del words[-1]
        
    return words

# returns all the matched bill names found in utterance
def find_bill_names(utterance):
    input_list = utterance.lower().split()
    pattern_list_list = []
    res = []

    for n in range(1,5):
        ngrams = (list(zip(*[input_list[i:] for i in range(n)])))

        if n == 4: 
            pattern_list_list = [bill_id_pattern_4_1, bill_id_pattern_4_2]

        elif n == 3:
            pattern_list_list = [bill_id_pattern_3_1, bill_id_pattern_3_2]

        elif n == 2:
            pattern_list_list = [bill_id_pattern_2_1, bill_id_pattern_2_2,
             bill_id_pattern_2_3, bill_id_pattern_2_4,
             bill_id_pattern_2_5, bill_id_pattern_2_6,
             bill_id_pattern_2_7, bill_id_pattern_2_8,
             bill_id_pattern_2_9, bill_id_pattern_2_10]

        elif n == 1:
            pattern_list_list = [bill_id_pattern_1_1, bill_id_pattern_1_2,
             bill_id_pattern_1_3, bill_id_pattern_1_4, bill_id_pattern_1_5,
             bill_id_pattern_1_6, bill_id_pattern_1_7, bill_id_pattern_1_8,
             bill_id_pattern_1_9, bill_id_pattern_1_10]
        
        for word in ngrams:
            target = ' '.join(word).strip()
            bill_names = find_any_pattern(target, pattern_list_list)

            if len(bill_names) > 0: res.append(bill_names)
    
    return res

# utterance text
def replace_bill_ids_in_utterance(utterance):
    # split text into list of words
    words = utterance.lower().split()
    utterance_length = len(words)

    word_ix = 0 # index into the utterance
    bill_id_replaced = False # if something has been replaced

    # going through each word
    while(word_ix < utterance_length):
        # match a four word pattern
        if (word_ix < (utterance_length-3) and
            matches_any_4_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2],
                                         words[word_ix+3])):

            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 3)
            utterance_length -= 3
            bill_id_replaced = True
            
        # match a three word pattern
        elif (word_ix < (utterance_length-2) and
              matches_any_3_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2])):
            
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 2)
            utterance_length -= 2
            bill_id_replaced = True
            
        # match a two word pattern
        elif (word_ix < (utterance_length-1) and
            matches_any_2_word_pattern(words[word_ix],
                                         words[word_ix+1])):
            
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 1)
            utterance_length -= 1
            bill_id_replaced = True
            
        # match a one word pattern
        elif (matches_any_1_word_pattern(words[word_ix])):
            words[word_ix] = "<BILL_ID>"
            bill_id_replaced = True
            

        word_ix += 1
            
    return (" ".join(words), bill_id_replaced)

def replace_bill_ids(old, new):
    for line in old:
        line_splits = line.lower().rstrip("\n").split("~")
        (new_text, bill_id_replaced) = replace_bill_ids_in_utterance(line_splits[2])
        new.write(line_splits[0] + "~" + line_splits[1] + "~" + new_text + "~" + line_splits[3] + "\n")