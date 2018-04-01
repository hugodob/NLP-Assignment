1.1
Answer is contained in file : candidate_hmm_postags
To run the script just execute: python hmm_postagging.py

1.2
Answer is contained in the files : candidate_parses and candidate_pcgf_postags
To run the script just execute : pcfg_parsing.py

1.3
Answer is contained in file : candidate_dual_decomp_postags
To run the script just execute : ./dual_decomposition_postags.py --hmm_trans=hmm_trans --hmm_emits=hmm_emits --pcfg=pcfg --candidate_sents=test_sents --sigma_init=1 --sigma_decay=0.7 --K=20

2
Just execute "python ner-constrained.py" to output the result with the perceptron
---- execute "python ner-unconstrained.py" to output the result with LGBM
