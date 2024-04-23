
# %%
BATCH_SIZE=["1", "4", "8", "16"]
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


def extract_time(time_file_path, bs):
    if os.path.isfile(time_file_path):
        # Get times for ceil(100/BATCH_SIZE) batches (100 samples with batc size bs)
        # We don't retrieve the first time because it's slightly slower than other times
        # We don't retrieve the last time because it's not precisely batch size bs
        get_time = tailer.tail(open(time_file_path), NUM_SAMPLE if args.task=="txt_to_img" else math.ceil(100/bs))[1:-1]
        assert len(get_time)>0
        if not all([t.replace(".", "").isnumeric() for t in get_time]):
            print("Didn't properly end :" , time_file_path)
        else:
            return [float(t) for t in get_time]
    else:
        print("File doesn't exist: " + time_file_path)
    return list()



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


BASE_DIR="/fsx-checkpoints/yejinlee/sweep/"+args.task+"/"+args.template+"."+args.model
RETRIEVAL_BASE_DIR="/fsx-checkpoints/yejinlee/sweep/"+args.task+"/"+args.retrieval_template+"."+args.model

for bs in BATCH_SIZE:
    EXP_DIR=BASE_DIR+"/bs"+bs+"/original/"
    time_file_path=EXP_DIR+"time.txt"
    xformer_baseline_time = np.average(extract_time(time_file_path, int(bs)))
    if "txt_to_img" in EXP_DIR:
        score_file_path=EXP_DIR+"score.txt"
        baseline_score, baseline_score_second = extract_score(score_file_path)
    else:
        score_file_path=EXP_DIR+"results/"+args.template+".json"
        baseline_score, baseline_score_second = extract_score_from_json(score_file_path)

    print("Batch Size ", bs, " Baseline time: ", xformer_baseline_time)
    print("Batch Size ", bs, " Baseline score: ", baseline_score)
    xaxis.append("xformer Original bs" + bs)
    time_result.append(xformer_baseline_time)
    score_result.append(0)

    for nd in NRETRIEVED_DOCS:
        time_file_path=RETRIEVAL_BASE_DIR+"/bs"+bs+".n_retrieved_doc"+nd+"/time.txt"
        time = extract_time(time_file_path, int(bs))                    
        
        if "txt_to_img" in BASE_DIR:
            score_file_path=RETRIEVAL_BASE_DIR+"/bs"+bs+".n_retrieved_doc"+nd+"/score.txt"
            score, score_second = extract_score(score_file_path)
        else:
            score_file_path=RETRIEVAL_BASE_DIR+"/bs"+bs+".n_retrieved_doc"+nd+"/results/"+args.retrieval_template+".json"
            score, score_second = extract_score_from_json(score_file_path)

        
        xaxis.append("bs"+bs+".n_retrieved_doc"+nd)
        time_result.append(np.average(time))
        score_result.append(score-baseline_score)

# %%
assert len(xaxis) == len(time_result)
fig, ax = plt.subplots(figsize=[25.6, 4.8])

ax2 = plt.twinx()
ax2.set_ylabel('Clip Score Loss')

from matplotlib import colormaps
colormap = colormaps['Set3'].colors


NUM_BATCH_CONFIG=len(BATCH_SIZE)
NUM_NRD_CONFIG=len(NRETRIEVED_DOCS)+1
for i in range(0, NUM_BATCH_CONFIG*NUM_NRD_CONFIG, NUM_NRD_CONFIG):
    ax.bar(xaxis[i:i+NUM_NRD_CONFIG], time_result[i:i+NUM_NRD_CONFIG], label="Batch Size "+str(BATCH_SIZE[int(i/NUM_NRD_CONFIG)]), color=colormap[int(i/NUM_NRD_CONFIG)])
    ax2.plot(xaxis[i:i+NUM_NRD_CONFIG], score_result[i:i+NUM_NRD_CONFIG], marker='*', color='palegreen', ms=5)#, label='Clip Score Loss')

ax2.plot([], [], marker='*', color='palegreen', ms=5, label='Clip Score Loss')

# Rotating X-axis labels
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel("Latency per Sample (ms)")

ax.legend(loc="lower right")
ax2.legend(loc="lower left")


# plt.title(BASE_DIR.split("/")[-1]+ "bs."+str(BATCH_SIZE)+' ToMe Breakdown (ms) & Clip Score Drop')
plt.title("base_"+args.template+"_retrieval_"+args.retrieval_template+"."+args.model+' Latency per Sample (ms) & '+ ('Clip' if args.task=="txt_to_img" else "Cider") + ' Score Loss')
plt.grid()

# Show plot
fig.tight_layout()
fig.show()
save_file_path="/fsx-checkpoints/yejinlee/analysis_figures/time_measure/"+args.task
os.makedirs(save_file_path, exist_ok=True)
save_file_path += "/base_"+args.template+"_retrieval_"+args.retrieval_template+".png"
print("Saving to ", save_file_path)


plt.savefig(save_file_path)
# %%

