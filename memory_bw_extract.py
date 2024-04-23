# %%
PROFILE_GRANULARITY=8
NUM_SAMPLE=6
PROFILE_LAYER=5

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import re
import sys
sys.path.insert(0, "/usr/local/cuda-12.1/nsight-compute-2023.1.1/extras/python")
import ncu_report

parser = argparse.ArgumentParser("argparser")
# Parse possible arguments:
parser.add_argument("--task", type=str, default="img_to_txt")
parser.add_argument("--profile_dir", type=str, default="")
args = parser.parse_args()


if "coco.0_shot.cm3v2_template" in args.profile_dir:
    END=13
elif "flickr30k.0_shot.cm3v2_template" in args.profile_dir:
    END=13
elif "coco_image.0_shot" in args.profile_dir:
    END=1023
elif "partiprompts.0_shot" in args.profile_dir:
    END=1023
elif "okvqa.0_shot.cm3v2_template" in args.profile_dir:
    END=4
elif "textvqa.0_shot.cm3v2_template" in args.profile_dir:
    END=8
elif "vizwiz.0_shot.cm3v2_template" in args.profile_dir:
    END=5

def process_report(file_path):
    report = ncu_report.load_report(file_path)
    mem = list()
    sm = list()
    for range_idx in range(report.num_ranges()):
        current_range = report.range_by_idx(range_idx)
        print(file_path, " ", len(current_range))

        # if len(current_range) < 230:
        #     print("Not enough profiling for ", file_path, " ", len(current_range))
        for fs in filter_ps:
            local_memory_bw_list = list()
            local_sm_util = list()
            for action_idx in current_range.actions_by_nvtx(["hello_"+str(fs)+"/"], []):
                action = current_range.action_by_idx(action_idx)
                # METRICS="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed"
                # print(action.name())
                local_memory_bw_list.append(action['dram__throughput.avg.pct_of_peak_sustained_elapsed'].value())
                local_sm_util.append(action['sm__throughput.avg.pct_of_peak_sustained_elapsed'].value())
            mem.append(np.average(local_memory_bw_list))
            sm.append(np.average(local_sm_util))
    return mem, sm

xaxis = list()
memory_bw_util = list()
sm_util = list()
for profile_step in tqdm(range(0, END+1, PROFILE_GRANULARITY)):
    profile_steps=""
    filter_ps = []
    for ps in range(profile_step, profile_step+PROFILE_GRANULARITY):
        if ps > END:
            break
        profile_steps += str(ps)+"."
        filter_ps.append(ps)
        xaxis.append("Decoding Step "+str(ps))
    # for fs in filter_ps:
    #     memory_bw_util.append(0)
    #     sm_util.append(0)
    # continue

    file_path = args.profile_dir+"/layer"+str(PROFILE_LAYER)+"/decoding_step"+profile_steps+"/ncu_memory_bw_profile.ncu-rep"
    
    if os.path.isfile(file_path):
        mem, sm = process_report(file_path)
        memory_bw_util += mem
        sm_util += sm
    else:
        file_path = args.profile_dir+"/layer"+str(PROFILE_LAYER)+"/decoding_step_"+profile_steps+"/ncu_memory_bw_profile.ncu-rep"
        if os.path.isfile(file_path):
            mem, sm = process_report(file_path)
            memory_bw_util += mem
            sm_util += sm
        else:
            if "cfg" in file_path:
                # file_path = args.profile_dir+"/layer"+str(PROFILE_LAYER)+"/decoding_step_"+profile_steps+"/ncu_memory_bw_profile.ncu-rep"
                file_path = re.sub(r'topp', 'topp.', re.sub(r'temp', 'temp.', re.sub(r'cfg', 'cfg.', args.profile_dir)))+"/layer"+str(PROFILE_LAYER)+"/decoding_step"+profile_steps+"/ncu_memory_bw_profile.ncu-rep"
                if os.path.isfile(file_path):
                    mem, sm = process_report(file_path)
                    memory_bw_util += mem
                    sm_util += sm
                else:
                    print("File doesn't exist ", file_path)
                    for fs in filter_ps:
                        memory_bw_util.append(0)
                        sm_util.append(0)
            else:
                print("File doesn't exist ", file_path)
                for fs in filter_ps:
                    memory_bw_util.append(0)
                    sm_util.append(0)


#%%
def plot_graph(xaxis, yaxis, title):
    assert len(xaxis) == len(yaxis), str(len(xaxis)) + " " + str(len(yaxis))
    fig, ax = plt.subplots(figsize=[200, 20] if len(xaxis)>1000 else [25.6, 4.8])

    # Creating plot
    ax.bar(xaxis, yaxis)
    ax.set_ylabel(title + " %")

    # Rotating X-axis labels
    ax.tick_params(axis='x', rotation=90)

    plt.title(args.profile_dir.split("/")[-1] + " " + title)
    plt.ylim(0,100)
    plt.grid()
    plt.margins(-0.0001)

    # Show plot
    fig.tight_layout()
    fig.show()
    save_file_path="/fsx-checkpoints/yejinlee/analysis_figures/"+"_".join(title.lower().split(" "))+"/"
    os.makedirs(save_file_path, exist_ok=True)
    save_file_path += args.profile_dir.split("/")[-1]+".png"
    print("Saving to ", save_file_path)
    plt.savefig(save_file_path)


#%%
plot_graph(xaxis, memory_bw_util, "Memory Bw Utilization")
plot_graph(xaxis, sm_util, "SM Utilization")
# %%