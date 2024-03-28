
# %%
TOME_R=[0.25, 0.5, 0.75, 0.95]
# TOME_LAYER=[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [14,15,16,17], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]]
# TOME_NUM_TOK=[34, 64, 128, 256, 384, 512, 640, 768, 896]
TOME_LAYER=([0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15], [16,17,18,19,20,21,22,23], [24,25,26,27,28,29,30,31], [0,1,2,3,4,5,6,7,8,9,10,11], [8,9,10,11,12,13,14,15,16,17,18,19], [12,13,14,15,16,17,18,19,20,21,22,23], [16,17,18,19,20,21,22,23,24,25,26,27], [20,21,22,23,24,25,26,27,28,29,30,31], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
TOME_NUM_TOK=[34]

TOME_MERGE_TIME=["True", "False"]
BATCH_SIZE=32
NUM_SAMPLE=6

import os
import tailer
import numpy as np
from contextlib import suppress
import matplotlib.pyplot as plt
import re

def extract_time(time_file_path):
    if os.path.isfile(time_file_path):
        # Get times for NUM_SAMPLE batches
        get_time = tailer.tail(open(time_file_path), NUM_SAMPLE)
        if not all([t.replace(".", "").isnumeric() for t in get_time]):
            print("Didn't properly end :" , time_file_path)
            # break
        else:
            return [float(t) for t in get_time[1:]]
    else:
        print("File doesn't exist: " + time_file_path)
        # break
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
        print("File doesn't exist: " + time_file_path)
        # break
    return -1, -1

xaxis = list()
time_result = {"Merge": list(), "Other": list()}
score_result = list()

BASE_DIR="/fsx-checkpoints/yejinlee/sweep/txt_to_img/coco_image.0_shot.cm3v21_109m_sft"
time_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/original/time.txt"
xformer_baseline_time = np.average(extract_time(time_file_path))
score_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/original/score.txt"
baseline_clip_score, baseline_pick_score = extract_score(score_file_path)
print("Baseline time: ", xformer_baseline_time)
print("Baseline clip score: ", baseline_clip_score)

for r in TOME_R:
    for l in TOME_LAYER:
        for nt in TOME_NUM_TOK:
            time = {"True":list(), "False":list()}
            clip_score = -1
            pick_score = -1
            for m in TOME_MERGE_TIME:
                time_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/r_"+str(r)+"_layer_"+str(l).replace(" ", "")+"_num_tok_"+str(nt)+"_merge_time_"+m+"/time.txt"
                time[m] = extract_time(time_file_path)                    
                
                if m == "False":
                    score_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/r_"+str(r)+"_layer_"+str(l).replace(" ", "")+"_num_tok_"+str(nt)+"_merge_time_"+m+"/score.txt"
                    clip_score, pick_score = extract_score(score_file_path)
  
            if len(time["True"])==0 and len(time["False"]) == 0:
                continue
            
            if len(time["True"]) == 0:
                merge_time = 0
            else:
                merge_time = np.average(time["True"])
            run_time = np.average(time["False"]) - merge_time

            xaxis.append("r_"+str(r)+"_layer_"+str(l).replace(" ", "")+"_num_tok_"+str(nt))
            time_result["Merge"].append(merge_time)
            time_result["Other"].append(run_time)
            score_result.append(clip_score)
            print(merge_time,  " ", run_time)


# %%
print(len(xaxis))
assert len(xaxis) == len(time_result["Merge"]) and len(time_result["Merge"]) == len(time_result["Other"])
fig, ax = plt.subplots(figsize=[25.6, 4.8])

xaxis.insert(0, "Xformer Original")
score_result.insert(0, baseline_clip_score)
time_result["Other"].insert(0, xformer_baseline_time)
time_result["Merge"].insert(0, 0)

# baseline_time = 56540.17701
# xaxis.insert(0, "Original")
# score_result.insert(0, baseline_clip_score)
# time_result["Other"].insert(0, baseline_time)
# time_result["Merge"].insert(0, 0)

# Creating plot
ax.bar(xaxis, time_result["Other"], label="Other")
ax.bar(xaxis, time_result["Merge"], label="Merge", bottom=time_result["Other"])
ax.axhline(y = xformer_baseline_time, color = 'r', linestyle = '-') 

ax2 = plt.twinx()
ax2.plot(xaxis, [s-baseline_clip_score for s in score_result], marker='*', color='palegreen', ms=5, label='Clip Score Loss')
ax2.set_ylabel('Clip Score Loss')


# Rotating X-axis labels
ax.tick_params(axis='x', rotation=90)

ax.legend(loc="lower right")
ax2.legend(loc="lower left")


plt.title(BASE_DIR.split("/")[-1]+ "bs."+str(BATCH_SIZE)+' ToMe Breakdown (ms) & Clip Score Drop')
plt.grid()

# Show plot
fig.show()
plt.savefig("./sweep.pdf")
# %%

