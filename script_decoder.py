
import numpy as np
import pandas as pd
import k2
import torch
import pickle
import sys
from math import floor
import struct
import time
import sentencepiece as spm
import jiwer as jw
import difflib as dl
import nltk
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


fsa_path = sys.argv[1]
ref_path = sys.argv[2]
model_path = sys.argv[3]

with open(fsa_path, 'rb') as file:
    fsa = k2.Fsa.from_dict(torch.load(file, map_location=torch.device('cpu')))

with open(ref_path, 'rb') as file:
    ref = pickle.load(file)
    


bpe_model = spm.SentencePieceProcessor()
bpe_model.load(model_path)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

 
# int32_to_float converts a number encoded as a signed 32bits integer to a float.


def int32_to_float(number):
    if(np.issubdtype(type(number), np.integer)):
        return struct.unpack('f', struct.pack('i', number))[0]
    else:
        return number


def compute_backward(fsa, index):
    tensor = torch.tensor([index], dtype=torch.int32)
    fsa_index = k2.index_fsa(fsa, tensor)
    back_scores = fsa_index._get_backward_scores(True,True)
    return back_scores.numpy()


def successors(fsa_list, cur_node, backward, factor):
    successors = []

    scores_AM = []
    scores_LM = []


    for node in fsa_list:
        if(node[0] == cur_node[1]):
            new_list = cur_node[2].copy()
            new_list.append(node[2])
            score = int32_to_float(cur_node[3]) + int32_to_float(node[3]) + backward[node[1]]
            scores_AM.append(score)
            #score_LM  = rescore_LLM_2(new_list, score, .1)
            #successors.append([cur_node[0], node[1], new_list, score_LM])
            #scores_LM.append(score_LM)
            successors.append([cur_node[0], node[1], new_list, score])
        else:
            continue

    return successors, scores_AM

 
# insertion_sort sorts any list of nodes.


def insertion_sort(fsa_list):
    for i in range(1, len(fsa_list)):
        key = fsa_list[i]
        j = i - 1
        while j >= 0 and key[3] > fsa_list[j][3]:
            fsa_list[j + 1] = fsa_list[j]
            j -= 1
        fsa_list[j + 1] = key
    return fsa_list

 
# Implementing the rescoring LLM function


def rescore_LLM(hyps):
    n_best_list = []
    scores_LM = []
    for hyp in hyps:
        sentence_tokens = hyp[2]
        scores = hyp[-1]
        ours = copy(decode(sentence_tokens))
        sentence = bpe_model.decode(ours)
        n_best_list.append({"node": hyp, "sentence": sentence, "score": scores})
    rescored_list = []
    for hypothesis in n_best_list:
        text = hypothesis["sentence"]
        input_ids = tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids)
        log_probabilities = outputs.logits[:, -1, :]
        # Assuming you are using log probabilities, you can transform it into probabilities if needed.
        new_score = hypothesis["score"] + log_probabilities[0, -1].item()
        scores_LM.append(new_score)
        rescored_list.append([hypothesis["node"][0], hypothesis["node"][1], hypothesis["node"][2], new_score])
    return rescored_list, scores_LM

 
# Rescores individual hypotheses


def rescore_LLM_2(hyp, score, coeff):
    ours = copy(decode(sentence_tokens))
    sentence = bpe_model.decode(ours)
    rescored_list = []
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    log_probabilities = outputs.logits[:, -1, :]
    # Assuming you are using log probabilities, you can transform it into probabilities if needed.
    new_score = score + log_probabilities[0, -1].item()*coeff

    t = time.time() - start
    return new_score

 
# Implementation of a stack decoder, using prunning in list of nodes to explore (open). Thus, we only explore the most promising path. Not taking the forward/backward functions into account yet.


def stack_decoder_partial(fsa, num, factor, threshold):

    backward = compute_backward(fsa, num)

    fsa = k2.to_tensor(fsa[num]).numpy() # convert fsa to numpy array
    open =[]
    closed = []

    start = fsa[0][0]
    end = fsa[-1][1]

    i = 0
    while(fsa[i][0] == start): # find all arcs that start at the start state
        open.append([fsa[i][0], fsa[i][1], [fsa[i][2]], fsa[i][3] + backward[fsa[i][1]]])
        i+=1
    
    i=0
    while(len(open) > 0):
        for cur_node in open:
            if(cur_node[2] == -1):
                return closed
            suc_nodes = successors(fsa, cur_node, backward, factor)
            suc_nodes = suc_nodes[0]
            open.remove(cur_node)
            closed.append(cur_node)
            open.extend(suc_nodes)
            insertion_sort(open)
            if(len(open) > 10):
                del_index = floor(len(open) * threshold)
                del open[-del_index:]
        i+=1

    rescored, scores_LM = rescore_LLM(closed[-5:])
    closed = insertion_sort(rescored)
    
    return closed

 
# Associates tokens to their related string and outpus the final sentence.


def decode(sentence_tokens):
    tokens = []

    for i in sentence_tokens:
        if i == -1:
            break
        elif i == 0:
            continue
        tokens.append(i)
    

    return tokens


def word_error_rate(ref, hyp):
    return jw.wer(ref, hyp)

 
# I obtained a really strange error: decode doesn't work with output token sequence defined by a function but does with the same output function defined as-is. I have to convert the list elements to strings and then back to ints.


def copy(list):
    list_str= []
    list_out= []

    for i in list: #Converts to string
        var = str(i)
        list_str.append(var)

    for i in list_str: #Converts back to ints
        var = int(i)
        list_out.append(var)
    return(list_out)


def wer(r, h):

    ref = r 

    r = nltk.word_tokenize(r)
    h = nltk.word_tokenize(h)

    """
    This function returns the word error rate (WER) given two lists of strings:
    r (reference) and h (hypothesis).
    """
    # Create a dictionary of words and their counts in the reference text
    ref_dict = {}
    for word in r:
        if word not in ref_dict:
            ref_dict[word] = 1
        else:
            ref_dict[word] += 1

    # Create a dictionary of words and their counts in the hypothesis text
    hyp_dict = {}
    for word in h:
        if word not in hyp_dict:
            hyp_dict[word] = 1
        else:
            hyp_dict[word] += 1

    # Calculate the number of substitutions, deletions, and insertions
    substitutions = 0
    deletions = 0
    insertions = 0
    for word in ref_dict.keys():
        if word not in hyp_dict:
            deletions += ref_dict[word]
        elif hyp_dict[word] < ref_dict[word]:
            substitutions += hyp_dict[word]
            deletions += ref_dict[word] - hyp_dict[word]
        elif hyp_dict[word] > ref_dict[word]:
            substitutions += ref_dict[word]
            insertions += hyp_dict[word] - ref_dict[word]
        else:
            substitutions += ref_dict[word]

    # Calculate the WER
    wer = (substitutions + deletions + insertions) / len(ref.split())

    return wer, substitutions, deletions, insertions



subs_ours = []
dels_ours = []
ins_ours = []
avg_time = []

wer_values_ours = []

def main():
    for i in range(len(ref)):
        start_time = time.time()
        decoded_partial = stack_decoder_partial(fsa, i, 1, .99)
        sentence_tokens = decoded_partial[-1][2]
        ours = copy(decode(sentence_tokens))
        ours_decoded = bpe_model.decode(ours)
        wer_score = wer(ours_decoded, ref[i])
        subs_ours.append(wer_score[1])
        dels_ours.append(wer_score[2])
        ins_ours.append(wer_score[3])
        wer_values_ours.append(wer_score[0])
        avg_time.append(time.time() - start_time)
        
    print("Average time: ", np.mean(avg_time))
    print("Substitutions: ", np.mean(subs_ours))
    print("Deletions: ", np.mean(dels_ours))
    print("Insertions: ", np.mean(ins_ours))
    print("WER :", np.mean(wer_values_ours))


main()


