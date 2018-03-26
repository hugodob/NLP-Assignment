import re
import io
import pandas as pd
import csv
import numpy as np

hmm_emits=pd.read_table('hmm_emits', header=None, quoting=csv.QUOTE_NONE)
hmm_emits.columns=["tag", "word", "log_prob"]
count=0
words=dict()
for word in hmm_emits["word"]:
    if(word not in words):
        words[word]=count
        count+=1
count=0
tags=dict()
for tag in hmm_emits["tag"]:
    if(tag not in tags):
        tags[tag]=count
        count+=1
nb_tags=len(tags)
nb_words=len(words)
emits_matrix=np.full((nb_tags,nb_words), -np.inf)
for i in range(len(hmm_emits)):
    index_i=tags[hmm_emits["tag"][i]]
    index_j=words[hmm_emits["word"][i]]
    emits_matrix[index_i][index_j]=hmm_emits["log_prob"][i]

hmm_trans=pd.read_table('hmm_trans', header=None, quoting=csv.QUOTE_NONE)
hmm_trans.columns=["source", "target", "log_prob"]


def run_viterbi(sequence, transition_scores, emission_scores):

    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(initial_scores)  # Number of states.

    # Variables storing the Viterbi scores.
    viterbi_scores = np.zeros([length, num_states]) + logzero()
    viterbi_scores[0,:]=emission_scores[0, :] + initial_scores
    # Variables storing the paths to backtrack.
    viterbi_paths = -np.ones([length, num_states], dtype=int)

    # Most likely sequence.
    best_path = -np.ones(length, dtype=int)

    # ----------
    for pos in xrange(1, length):
        for current_state in xrange(num_states):
            viterbi_scores[pos][current_state]= np.max(viterbi_scores[pos-1, :]+transition_scores[pos-1, current_state, :])+emission_scores[pos, current_state]
            viterbi_paths[pos][current_state]=np.argmax(viterbi_scores[pos-1, :]+transition_scores[pos-1, current_state, :])
    best_path[length-1]=np.argmax(viterbi_scores[length-1,:]+final_scores[:])
    best_score=np.max(final_scores[:]+viterbi_scores[length-1,:])
    for pos in xrange(length-2, -1, -1):
        best_path[pos]=viterbi_paths[pos+1][best_path[pos+1]]
    return best_path, best_score
