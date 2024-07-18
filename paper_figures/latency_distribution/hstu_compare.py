# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
from statistics import geometric_mean
from matplotlib import colormaps


# DIR_PREFIX=""
# DIR_PREFIX="./onellm_scripts/data_for_paper/latency_dist/"
DIR_PREFIX="/Users/yejinlee/hpca_2025/onellm_scripts/data_for_paper/radar_chart/"

def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return [a for i, a in enumerate(data) if s[i] < m ]

def get_latency(file_path):
    file_path += "/timer_result.txt"
    timer_result = dict()
    if os.path.isfile(file_path):
        f = open(file_path, "r")
        print("Reading from ", file_path)
        headers = None
        for idx, sl in enumerate(f):
            if idx==0:
                headers = re.sub("\n", "", sl).split("\t")
                for h in headers:
                    timer_result[h] = list()
            else:
                if "Total" in sl and idx%2==0:
                    continue

                slsl = [float(s) for s in re.sub("\n", "", sl).split("\t") if s!=""]
                for idx, h in enumerate(headers):
                    timer_result[h].append(slsl[idx])
        # for k, v in timer_result.items():
        #     # timer_result[k].sort(reverse=True)
        #     timer_result[k] = reject_outliers(v)
    else:
        print("File doesn't exist: " + file_path)
    return timer_result

def get_folder(dataset, bs):
    if dataset == "HSTU-Triton":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/latency_distribution/batch_size_"+str(bs)+"/"
    elif dataset == "HSTU-PyTorch":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/latency_distribution/pytorch/batch_size_"+str(bs)+"/"
    else:
        print(dataset)
        assert False


def plot():
    batch_size = [1,2,4,8,16,32,64,128,256,512]
    baseline = []
    for bs in batch_size:
        baseline.append(sum([np.average(v) for v in get_latency(get_folder("HSTU-PyTorch", bs)).values()]))

    baseline_w_sdpa = []
    for bs in batch_size:
        baseline_w_sdpa.append(sum([np.average(v) for v in get_latency(get_folder("HSTU-Triton", bs)).values()]))

    baseline_w_sdpa = [a/b if b>0 else (2 if a>0 else 0) for a, b in zip(baseline, baseline_w_sdpa)]

    baseline = [b/b if b>0 else 0 for b in baseline]

    print(baseline_w_sdpa)
    print("Geomean speedup: ", geometric_mean([b for b in baseline_w_sdpa if b!=0]))
    
    # create data 
    x = np.arange(len(baseline)) 
    width = 0.2

    fig, ax = plt.subplots(1, figsize=(8, 3), layout='tight')
    # plt.bar(x-0.1, baseline, width, color='cyan') 
    # plt.bar(x+0.1, baseline_w_sdpa, width, color='green') 
    plt.bar(x, baseline_w_sdpa, width, color=colormaps['Set3'].colors[4])

    for i in range(len(baseline)):
        if baseline[i]==0:
            plt.text(i-0.1, 0.3, 'Baseline w/o SDPA\ngives Out of Memory', ha='left', fontsize=8, rotation=90)


    plt.xticks(x, batch_size)#, rotation=90)
    # plt.ylim(0, 2) 
    plt.xlabel("Workloads") 
    plt.ylabel("Normalized Speedup") 
    # plt.title("Batch Size" + str(batch_size_dict["MSCOCO-34B"]))
    # plt.legend(["w/o SDPA", "w/ SDPA"], ncol=2, bbox_to_anchor=(0.5, 1.15), loc="upper center")
    
    plt.grid(lw=0.2)
    plt.ylim(1, 10) 
    dump_dir = './onellm_scripts/analysis_figures/latency_distribution/hstu/'
    os.makedirs(dump_dir, exist_ok=True)
    plt.savefig(dump_dir+"hstu.pdf", bbox_inches = 'tight')
    print("Saving to ", dump_dir+"hstu.pdf")
    plt.show() 



plot()
# %%
