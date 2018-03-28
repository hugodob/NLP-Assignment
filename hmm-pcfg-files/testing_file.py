import subprocess

def run_test(sigma_init, sigma_decay, K):
    command1="./dual_decomposition_postags.py --hmm_trans=hmm_trans --hmm_emits=hmm_emits --pcfg=pcfg --candidate_sents=dev_sents --sigma_init="
    command1+=str(sigma_init)+" --sigma_decay="+str(sigma_decay)+ " --K="+str(K)
    print(command1)
    command2="./eval.py --reference_postags_filename = dev_postags --candidate_postags_filename=candidate_dual_decomp_postags"

    ans1 = subprocess.check_output(command1).decode()
    ans2 = subprocess.check_output(command2).decode()
    print(ans2)

run_test(1,0.7,20)
