
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
parser.add_argument("--task", type=str, default="coco.0_shot.cm3v2_template")
parser.add_argument("--model", type=str, default="cm3v21_109m_sft")
args = parser.parse_args(args=[])


def extract_time(time_file_path, bs):
    if os.path.isfile(time_file_path):
        # Get times for ceil(100/BATCH_SIZE) batches (100 samples with batc size bs)
        # We don't retrieve the first time because it's slightly slower than other times
        # We don't retrieve the last time because it's not precisely batch size bs
        get_time = tailer.tail(open(time_file_path), math.ceil(100/bs))[:-1]
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
    return -1, -1

def extract_score_from_json(score_file_path):
    if os.path.isfile(score_file_path):
        file = open(score_file_path)
        return json.load(file)["cider_score"], -1
    else:
        print("File doesn't exist: " + score_file_path)
        # break
    return -1, -1

xaxis = list()
time_result = list()
score_result = list()


BASE_DIR="/fsx-checkpoints/yejinlee/sweep/img_to_txt/"+args.task+"."+args.model

for bs in BATCH_SIZE:
    EXP_DIR=BASE_DIR+"/bs"+bs+"/original/"
    time_file_path=EXP_DIR+"time.txt"
    xformer_baseline_time = np.average(extract_time(time_file_path, int(bs)))
    if "txt_to_img" in EXP_DIR:
        score_file_path=EXP_DIR+"score.txt"
        clip_score, pick_score = extract_score(score_file_path)
    else:
        score_file_path=EXP_DIR+"results/"+args.task+".json"
        baseline_clip_score, baseline_pick_score = extract_score_from_json(score_file_path)

    print("Batch Size ", bs, " Baseline time: ", xformer_baseline_time)
    print("Batch Size ", bs, " Baseline clip score: ", baseline_clip_score)
    xaxis.append("xformer Original bs" + bs)
    time_result.append(xformer_baseline_time)
    score_result.append(baseline_clip_score)

    for nd in NRETRIEVED_DOCS:
        time_file_path=BASE_DIR+"/retrieval/bs"+bs+".n_retrieved_doc"+nd+"/time.txt"
        time = extract_time(time_file_path, int(bs))                    
        
        if "txt_to_img" in BASE_DIR:
            score_file_path=BASE_DIR+"/retrieval/bs"+bs+".n_retrieved_doc"+nd+"/score.txt"
            clip_score, pick_score = extract_score(score_file_path)
        else:
            score_file_path=BASE_DIR+"/retrieval/bs"+bs+".n_retrieved_doc"+nd+"/results/"+args.task+".json"
            clip_score, pick_score = extract_score_from_json(score_file_path)

        
        xaxis.append("bs"+bs+".n_retrieved_doc"+nd)
        time_result.append(np.average(time))
        score_result.append(clip_score)
        print(time,  " ", clip_score)


# baseline_time = 56540.17701
# xaxis.insert(0, "Original")
# score_result.insert(0, baseline_clip_score)
# time_result["Other"].insert(0, baseline_time)
# time_result["Merge"].insert(0, 0)


# %%
assert len(xaxis) == len(time_result)
fig, ax = plt.subplots(figsize=[25.6, 4.8])

# Creating plot
ax.bar(xaxis, time_result, label="Latency/Sample")
# ax.bar(xaxis, time_result["Merge"], label="Merge", bottom=time_result["Other"])
# ax.axhline(y = xformer_baseline_time, color = 'r', linestyle = '-') 

ax2 = plt.twinx()
ax2.plot(xaxis, [s-baseline_clip_score for s in score_result], marker='*', color='palegreen', ms=5, label='Clip Score Loss')
ax2.set_ylabel('Clip Score Loss')


# Rotating X-axis labels
ax.tick_params(axis='x', rotation=90)

ax.legend(loc="lower right")
ax2.legend(loc="lower left")


# plt.title(BASE_DIR.split("/")[-1]+ "bs."+str(BATCH_SIZE)+' ToMe Breakdown (ms) & Clip Score Drop')
plt.title('.'.join(BASE_DIR.split("/")[-2:])+' Latency per Sample (ms) & Clip Score Loss')
plt.grid()

# Show plot
fig.show()
plt.savefig("./sweep.pdf")
# %%

