import re
import io
import pandas as pd
import csv
import numpy as np

hmm_emits=pd.read_table('hmm_emits', header=None, quoting=csv.QUOTE_NONE)
hmm_emits.columns=["tag", "word", "log_prob"]
#On fait un dico mot->index
count=0
words=dict()
for word in hmm_emits["word"]:
    if(word not in words):
        words[word]=count
        count+=1

with open("dev_sents", "r") as f:
    reader = csv.reader(f)
    sequences  = list(reader)
sequences = [element[0].split(" ") for element in sequences]
for i in range(len(sequences)):
    for j in range(len(sequences[i])):
        #On rajoute les mots inconnus
        if(sequences[i][j] not in words):
            words[sequences[i][j]]=count
            count+=1
        sequences[i][j]=words[sequences[i][j]]

#On fait un dico tag->index
count=0
tags=dict()
for tag in hmm_emits["tag"]:
    if(tag not in tags):
        tags[tag]=count
        count+=1
tags['sentence_boundary']=count
nb_tags=len(tags)
nb_words=len(words)

#On cree la matrice des scores d'émission: emits_matrix[i][j] est la log prob p(word=j|tag=i)
emits_matrix=np.full((nb_tags,nb_words), -20.) #On choisit -20 comme minimum car c'est ce qui nous donne la meilleure précision
for i in range(len(hmm_emits)):
    index_i=tags[hmm_emits["tag"][i]]
    index_j=words[hmm_emits["word"][i]]
    emits_matrix[index_i][index_j]=hmm_emits["log_prob"][i]

#On cree la matrice des scores de transmission: trans_matrix[i][j] est la log prob p(tag=j|tag_prec=i)
hmm_trans=pd.read_table('hmm_trans', header=None, quoting=csv.QUOTE_NONE)
hmm_trans.columns=["source", "target", "log_prob"]
trans_matrix=np.full((nb_tags,nb_tags), -20.)
for i in range(len(hmm_trans)):
    index_i=tags[hmm_trans["source"][i]]
    index_j=tags[hmm_trans["target"][i]]
    trans_matrix[index_i][index_j]=hmm_trans["log_prob"][i]

#On isole initial_scores (log(p(tag|start))) et final_scores ((log(p(tag|end))))
index_boundary=tags["sentence_boundary"]
initial_scores=trans_matrix[index_boundary,:]
final_scores=trans_matrix[:,index_boundary]




def run_viterbi(sequence, initial_scores, trans_matrix, final_scores, emits_matrix):

    length = len(sequence)  # Length of the sequence.
    nb_tags = emits_matrix.shape[0]  # Number of tags.

    # Variables storing the Viterbi scores.
    viterbi_scores = np.zeros([length, nb_tags]) -20.
    viterbi_scores[0,:]=emits_matrix[:,sequence[0]] + initial_scores[:]
    # Variables storing the paths to backtrack.
    viterbi_paths = -np.ones([length, nb_tags], dtype=int)

    # Most likely sequence.
    best_path = -np.ones(length, dtype=int)

    # ----------
    for pos in xrange(1, length):
        for current_state in xrange(nb_tags):
            viterbi_scores[pos][current_state]= np.max(viterbi_scores[pos-1, :]+trans_matrix[:, current_state])+emits_matrix[current_state, sequence[pos]]
            viterbi_paths[pos][current_state]=np.argmax(viterbi_scores[pos-1, :]+trans_matrix[:, current_state])
    best_path[length-1]=np.argmax(viterbi_scores[length-1,:]+final_scores[:])
    best_score=np.max(final_scores[:]+viterbi_scores[length-1,:])
    for pos in xrange(length-2, -1, -1):
        best_path[pos]=viterbi_paths[pos+1][best_path[pos+1]]
    print best_score
    return best_path


index_to_tags={v: k for k, v in tags.items()} #inverting tags dico
best_paths=[]
for sequence in sequences:
    path=run_viterbi(sequence, initial_scores, trans_matrix, final_scores, emits_matrix)
    word_path=[]
    for i in range(len(path)):
        word_path.append(index_to_tags[path[i]])
    best_paths.append(word_path)

with open('candidate_postags','w') as f:
         for s in best_paths:
             for item in s:
                 f.write("%s " % item)
             f.write("\n")
