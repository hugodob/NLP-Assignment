#!/usr/bin/env python
import re
import io
import argparse
import pandas as pd
import csv
import numpy as np
import time

################## READING THE ARGUMENTS
argParser = argparse.ArgumentParser(description="evaluates parses and/or POS-taggings")
argParser.add_argument("--hmm_trans", type=str, default='')
argParser.add_argument("--hmm_emits", type=str, default='')
argParser.add_argument("--pcfg", type=str, default='')
argParser.add_argument("--candidate_sents", type=str, default='')
argParser.add_argument("--sigma_init", type=float, default=1)
argParser.add_argument("--sigma_decay", type=float, default=0.7)
argParser.add_argument("--K", type=int, default=20)
args = argParser.parse_args()

################### SEQUENCES

with open(args.candidate_sents, "r") as f:
    reader = csv.reader(f)
    sequences  = list(reader)

sequences = [element[0].split(" ") for element in sequences]


######################## FOR PCFG ###############################
###################BUILDING A WORD DICT

count=0
words=dict()
for i in range(len(sequences)):
    for j in range(len(sequences[i])):
        #On rajoute les mots inconnus
        if(sequences[i][j] not in words):
            words[sequences[i][j]]=count
            count+=1
        # sequences[i][j]=words[sequences[i][j]]


################### BUILDING A PCFG DICT
category_names = [] #list of the name of all categories
pcfg_table = dict() #keys are tuple (context, decision[0], decision[1]), val are probas
with open(args.pcfg) as f:
    for line in f:
        line = line.split("\t")
        category_names.append(line[0])
        score = float(line[-1][:-1]) #get rid of the \n
        decision = line[1].split(" ")
        if len(decision)==1:
            pcfg_table[(line[0], decision[0])] = score
        else:
            pcfg_table[(line[0], decision[0], decision[1])] = score

category_names = list(set(category_names)) #for uniqueness
category_names.append('sentence_boundary')

ante_double = dict() #keys are (decision[0], decision[1]), val are list of contexts
                     #that exist for those decisions
for key in pcfg_table:
    if len(key)==3:
        if (key[1],key[2]) not in ante_double:
            ante_double[(key[1],key[2])] = [key[0]]
        else:
            ante_double[(key[1],key[2])].append(key[0])


def run_CKY(sequence, category_names, pcfg_table, u):

    N = len(sequence)

    #same as in the nlp_lab_3
    delta_table = [[dict() for j in range(N)] for i in range(N)]
    phi_table = [[dict() for j in range(N)] for i in range(N)]

    #initialization
    for i in range(N):
        for Z in category_names:
            if (Z,sequence[i]) in pcfg_table:
                delta_table[i][i][Z] = pcfg_table[(Z,sequence[i])]
                delta_table[i][i][Z] += u[i][Z]
        if len(delta_table[i][i])==0: #words that are not known
            for Z in category_names:
                delta_table[i][i][Z] = -20
                delta_table[i][i][Z] += u[i][Z]

    #recursive phase : I changed the order to make it go faster
    for i in range(2, N+1):
        for j in range(N-i+1):

            #concurrents_score[Z] : all possible score for delta_table[j][j+i-1][Z]
            #concurrents[Z] : contains the corresponding (X,Y,k)
            #At the end, we will compute max (and argmax) of those arrays
            concurrents_score = dict()
            concurrents = dict()

            for k in range(j+1,j+i):
                for X in delta_table[j][k-1]:
                    for Y in delta_table[k][j+i-1]:
                        if (X,Y) in ante_double:
                            for Z in ante_double[(X,Y)]:
                                if Z not in concurrents_score:
                                    concurrents_score[Z] = []
                                    concurrents[Z] = []
                                concurrents_score[Z].append(
                                    delta_table[j][k-1][X]+
                                    delta_table[k][j+i-1][Y]+pcfg_table[(Z,X,Y)])
                                concurrents[Z].append((X,Y,k))
            #find the best concurrent, i.e. the argmax phase for every possible Z
            for Z in concurrents:
                max_index = 0
                for index in range(len(concurrents_score[Z])):
                    if concurrents_score[Z][index]>concurrents_score[Z][max_index]:
                        max_index = index
                delta_table[j][j+i-1][Z] = concurrents_score[Z][max_index]
                phi_table[j][j+i-1][Z] = concurrents[Z][max_index]

    def compute_POS(a,b,Z):
        '''recursive function for the backtracking'''
        if a==b:
            return Z+" "
        else:
            s = ""
            X,Y,k = phi_table[a][b][Z]
            s += compute_POS(a, k-1, X)
            s += compute_POS(k, b, Y)
            return s

    if "S" in delta_table[0][N-1]:
        return compute_POS(0, N-1, "S")
    else:
        return "err"


#################################### END PCFG ###############################

#################################### HMM part ###############################
hmm_emits=pd.read_table(args.hmm_emits, header=None, quoting=csv.QUOTE_NONE)
hmm_emits.columns=["tag", "word", "log_prob"]
hmm_trans=pd.read_table(args.hmm_trans, header=None, quoting=csv.QUOTE_NONE)
hmm_trans.columns=["source", "target", "log_prob"]

tags=[]
for tag in hmm_emits["tag"]:
    if(tag not in tags):
        tags.append(tag)
tags.append('sentence_boundary')


emits_matrix=[dict(zip(tags, [-20. for i in range(len(tags))])) for j in range(len(words))]
for i in range(len(hmm_emits)):
    if(hmm_emits["word"][i] in words and hmm_emits["tag"][i] in tags):
        emits_matrix[words[hmm_emits["word"][i]]][hmm_emits["tag"][i]]=hmm_emits["log_prob"][i]

trans_matrix=[dict(zip(tags, [-20. for i in range(len(tags))])) for j in range(len(tags))]
for i in range(len(hmm_trans)):
    trans_matrix[tags.index(hmm_trans["source"][i])][hmm_trans["target"][i]]=hmm_trans["log_prob"][i]

#On isole initial_scores (log(p(tag|start))) et final_scores ((log(p(tag|end))))
index_boundary=len(tags)-1
initial_scores=trans_matrix[index_boundary]
final_scores=[trans_matrix[i]['sentence_boundary'] for i in range(len(tags))]

def run_HMM(sequence, initial_scores, trans_matrix, final_scores, emits_matrix, u):

    length = len(sequence)  # Length of the sequence.

    # Variables storing the Viterbi scores.
    viterbi_scores = [dict(zip(tags, [-20. for i in range(len(tags))])) for j in range(len(sequence))]
    for tag in tags:
        viterbi_scores[0][tag]=emits_matrix[words[sequence[0]]][tag] + initial_scores[tag]-u[0][tag]
    # Variables storing the paths to backtrack.
    viterbi_paths = -np.ones([length, len(tags)], dtype=int)

    # Most likely sequence.
    best_path = -np.ones(length, dtype=int)

    # ----------
    for pos in range(1, length):
        for current_tag in tags:
            viterbi_scores[pos][current_tag]= np.max([viterbi_scores[pos-1][tags[i]]+trans_matrix[i][current_tag] for i in range(len(tags))])+emits_matrix[words[sequence[pos]]][current_tag]-u[pos][current_tag]
            viterbi_paths[pos][tags.index(current_tag)]=np.argmax([viterbi_scores[pos-1][tags[i]]+trans_matrix[i][current_tag] for i in range(len(tags))])
    best_path[length-1]=np.argmax([viterbi_scores[length-1][tags[i]]+final_scores[i] for i in range(len(tags))])
    best_score=np.max([viterbi_scores[length-1][tags[i]]+final_scores[i]for i in range(len(tags))])
    for pos in range(length-2, -1, -1):
        best_path[pos]=viterbi_paths[pos+1][best_path[pos+1]]
    s=''
    for i in best_path:
        s+=tags[i]
        s+=' '
    return s

##################################### END HMM ###############################

#the 3 hyper_parameters
sigma_init = args.sigma_init
sigma_decay = args.sigma_decay
K = args.K

def compute_postags(sequence, category_names, pcfg_table):
    u = [dict(zip(category_names, [0 for i in range(len(category_names))])) for j in range(len(sequence))]
    sigma_k=sigma_init
    for k in range(K):
        print("k = ", k)
        sigma_k *= sigma_decay
        # print("HMM running")
        y=run_HMM(sequence, initial_scores, trans_matrix, final_scores, emits_matrix, u)
        # print("CKY running")
        z=run_CKY(sequence, category_names, pcfg_table,u)
        y = y.split(" ")
        z = z.split(" ")
        flag_egual = True
        for i in range(len(y)):
            if y[i]!=z[i]:
                # print("diff at ",i)
                flag_egual = False
                u[i][y[i]] += sigma_k
                u[i][z[i]] -= sigma_k
        if flag_egual:
            break
    return run_HMM(sequence, initial_scores, trans_matrix, final_scores, emits_matrix, u)

best_paths = []
c = 0
for sequence in sequences:
    print(c,"/",len(sequence))
    c +=1
    path=compute_postags(sequence, category_names, pcfg_table)
    best_paths.append(path)


with open('candidate_dual_decomp_postags','w') as f:
    for s in best_paths:
        f.write("%s " % s)
        f.write("\n")
