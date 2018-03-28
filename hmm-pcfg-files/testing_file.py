import subprocess

def run_test(sigma_init, sigma_decay, K):
    command1="./dual_decomposition_postags.py --hmm_trans=hmm_trans --hmm_emits=hmm_emits --pcfg=pcfg --candidate_sents=dev_sents --sigma_init="
    command1+=str(sigma_init)+" --sigma_decay="+str(sigma_decay)+ " --K="+str(K)
    command1=command1.split(" ")
    command2="./eval.py --reference_postags_filename = dev_postags --candidate_postags_filename=candidate_dual_decomp_postags"
    command2=command2.split(" ")
    ans1 = subprocess.check_output(command1).decode()
    ans2 = subprocess.check_output(command2).decode()
    print(ans2)
    
sigma_init=[0.01, 0.1, 0.5, 1, 2]
for s in sigma_init:
    run_test(s,0.7,20)
