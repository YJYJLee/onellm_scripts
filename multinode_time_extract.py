
# %%
BATCH_SIZE=["1", "4", "8", "16", "32", "64", "128"]
NRETRIEVED_DOCS=["1", "2", "3", "4"]
NUM_SAMPLE=6


import os
import tailer
import numpy as np
from contextlib import suppress
import matplotlib.pyplot as plt
import re
import argparse
import json
import math
parser = argparse.ArgumentParser("argparser")

# Parse possible arguments:
parser.add_argument("--task", type=str, default="img_to_txt")
parser.add_argument("--template", type=str, default="coco.0_shot.cm3v2_template")
parser.add_argument("--retrieval-template", type=str, default="coco.0_shot.flamingo_retrieval_v2_template")
parser.add_argument("--model", type=str, default="cm3v21_109m_sft")
# args = parser.parse_args(args=[])
args = parser.parse_args()

parse_string = "ncs." if args.task == "txt_to_img" else "mbs."


def extract_time(time_file_path, bs, num_device):
    if os.path.isfile(time_file_path):
        # Get times for ceil(100/BATCH_SIZE) batches (100 samples with batc size bs)
        # We don't retrieve the first time because it's slightly slower than other times
        # We don't retrieve the last time because it's not precisely batch size bs
        get_time = tailer.tail(open(time_file_path), num_device)
        assert len(get_time)>0
        return np.average([float(gt.split("\t")[5]) for gt in get_time])
        # if not all([t.replace(".", "").isnumeric() for t in get_time]):
        #     print("Didn't properly end :" , time_file_path)
        # else:
            # return [float(t) for t in get_time]
    else:
        print("File doesn't exist: " + time_file_path)
    return 0



def extract_score(score_file_path):
    if os.path.isfile(score_file_path):
        found = False
        with open(score_file_path) as origin_file:
            for line in origin_file:
                found_line = re.findall(r'clip_score_strategy', line)
                if found_line:
                    get_clip_score_line = re.split(r"'clip_score': | 'pick_score': |}|,", line)[1:-1]
                    get_score = [float(get_clip_score_line[0]), float(get_clip_score_line[2])]
                    found = True

        if not found:
            print("Didn't properly end :" , score_file_path)
            # break
        else:
            # get_score = [float(s) for s in get_score]
            # clip_score = np.mean(get_score[:5])
            # pick_score = np.mean(get_score[5:])
            return get_score[0], get_score[1]   # clip score, pick score
    else:
        print("File doesn't exist: " + score_file_path)
        # break
    return np.nan, np.nan

def extract_score_from_json(score_file_path):
    if os.path.isfile(score_file_path):
        file = open(score_file_path)
        return json.load(file)["f1" if args.task == "img_txt_to_txt" else "cider_score"], -1
    else:
        print("File doesn't exist: " + score_file_path)
        # break
    return -1, -1

xaxis = list()
time_result = list()
score_result = list()


NGPU_NNODE=[(1, 1), (4, 1), (8, 1), (4, 2)]

for config in NGPU_NNODE:
    ngpu, nnode = config
    BASE_DIR="/fsx-atom/yejinlee/sweep/"+args.task+"/multigpu/"+str(ngpu)+"gpu_"+str(nnode)+"node/"
    xaxis.append(str(ngpu)+"gpu_"+str(nnode)+"node")
    for bs in BATCH_SIZE:
        split = args.template.split(parse_string)
        EXP_DIR=BASE_DIR+split[0]+parse_string+str(bs)+"."+".".join(split[1].split(".")[1:])
        time_file_path=EXP_DIR+"/timer_result.txt"
        time_result.append(np.average(extract_time(time_file_path, int(bs), ngpu*nnode)))


# %%
# assert len(xaxis) == len(time_result)
fig, ax = plt.subplots(figsize=[9.6, 4.8])

from matplotlib import colormaps
colormap = colormaps['Set3'].colors

x = np.arange(len(NGPU_NNODE))
shift = 0.1

print(len(time_result))

for idx, bs in enumerate(BATCH_SIZE):
    ax.bar([xx-shift*(3-idx) if idx<=3 else xx+shift*(idx-3) for xx in x], time_result[idx::len(BATCH_SIZE)], width=0.1, color=colormap[idx], label="Batch Size "+str(bs))

plt.xticks([val for pair in zip([xx-shift*3 for xx in x], [xx-shift*2 for xx in x], [xx-shift*1 for xx in x], [xx-shift*0 for xx in x], [xx+shift*1 for xx in x], [xx+shift*2 for xx in x], [xx+shift*3 for xx in x]) for val in pair], BATCH_SIZE*len(NGPU_NNODE), fontsize=6)

sec = ax.secondary_xaxis(location=0)
sec.set_xticks(x, labels=["\n\n"+xa for xa in xaxis], fontsize=6)#, rotation=90)


# Rotating X-axis labels
ax.tick_params(axis='x')#, rotation=90)
ax.set_ylabel("End-to-End Inference Runtime (ms)")

ax.legend(loc="upper left")
plt.title(args.template)
plt.grid(lw=0.2)

# Show plot
fig.tight_layout()
fig.show()
save_file_path="/fsx-atom/yejinlee/analysis_figures/time_measure/multigpu/"+args.task
os.makedirs(save_file_path, exist_ok=True)

split = args.template.split("/")[0].split(parse_string) if args.task == "txt_to_img" else args.template.split(parse_string) 
save_file_path += "/"+split[0]+".".join(split[1].split(".")[1:])+".pdf"
print("Saving to ", save_file_path)


plt.savefig(save_file_path)
# %%

