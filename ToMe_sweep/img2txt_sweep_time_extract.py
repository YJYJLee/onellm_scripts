
# %%
TOME_R=[0.25, 0.5, 0.75, 0.95]
TOME_LAYER=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [14, 15, 16, 17], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
TOME_NUM_TOK=[34, 64, 128, 256, 384, 512, 640, 768, 896]

# TOME_LAYER=[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [14,15,16,17], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]]
# TOME_NUM_TOK=[773]
# TOME_MERGE_TIME=["False"]

TOME_MERGE_TIME=["True", "False"]
BATCH_SIZE=32


import os
import tailer
import numpy as np
from contextlib import suppress
import matplotlib.pyplot as plt
import json
import math

xaxis = list()
time_result = {"Merge": list(), "Other": list()}
score_result = list()
task="coco.0_shot.cm3v2_template"

def extract_time(time_file_path):
    if os.path.isfile(time_file_path):
        # Get times for ceil(100/BATCH_SIZE) batches (100 samples with batc size BATCH_SIZE)
        # We don't retrieve the first time because it's slightly slower than other times
        get_time = tailer.tail(open(time_file_path), math.ceil(100/BATCH_SIZE))
        if not all([t.replace(".", "").isnumeric() for t in get_time]):
            print("Didn't properly end :" , time_file_path)
        else:
            return [float(t) for t in get_time]
    else:
        print("File doesn't exist: " + time_file_path)
    return list()

def extract_score(score_file_path):
    if os.path.isfile(score_file_path):
        file = open(score_file_path)
        return json.load(file)["cider_score"]
    else:
        print("File doesn't exist: " + score_file_path)
        # break
    return -1

BASE_DIR="/fsx-checkpoints/yejinlee/sweep/img_to_txt/coco.0_shot.cm3v2_template.cm3v21_109m_sft"
time_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/original/time.txt"
xformer_baseline_time = np.average(extract_time(time_file_path))
score_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/original/results/"+task+".json"
baseline_cider_score = extract_score(score_file_path)
print("Baseline time: ", xformer_baseline_time)
print("Baseline cider score: ", baseline_cider_score)


for r in TOME_R:
    for l in TOME_LAYER:
        for nt in TOME_NUM_TOK:
            time = {"True":list(), "False":list()}
            cider_score = -1
            pick_score = -1
            for m in TOME_MERGE_TIME:
                time_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/r_"+str(r)+"_layer_"+str(l).replace(" ", "")+"_num_tok_"+str(nt)+"_merge_time_"+m+"/time.txt"
                time[m] = extract_time(time_file_path)
                
                if m == "False":
                    score_file_path=BASE_DIR+"/bs"+str(BATCH_SIZE)+"/r_"+str(r)+"_layer_"+str(l).replace(" ", "")+"_num_tok_"+str(nt)+"_merge_time_"+m+"/results/"+task+".json"
                    cider_score = extract_score(score_file_path)

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
            score_result.append(cider_score)
            print(merge_time,  " ", run_time)

# %%
xaxis.insert(0, "Xformer Original")
score_result.insert(0, baseline_cider_score)
time_result["Other"].insert(0, xformer_baseline_time)
time_result["Merge"].insert(0, 0)
# baseline_cider_score = 0.832418
# xformer_baseline_time = 1989.770563
# baseline_time = 2323.633094
# xaxis.insert(0, "Original")
# score_result.insert(0, baseline_cider_score)
# time_result["Other"].insert(0, baseline_time)
# time_result["Merge"].insert(0, 0)



# %%
print(len(xaxis))
assert len(xaxis) == len(time_result["Merge"]) and len(time_result["Merge"]) == len(time_result["Other"])
fig, ax = plt.subplots(figsize=[128, 4.8])

# Creating plot
ax.bar(xaxis, time_result["Other"], label="Other")
ax.bar(xaxis, time_result["Merge"], label="Merge", bottom=time_result["Other"])
ax.axhline(y = xformer_baseline_time, color = 'r', linestyle = '-') 

ax2 = plt.twinx()
ax2.plot(xaxis, [(s-baseline_cider_score)*100 for s in score_result], marker='*', color='palegreen', ms=5, label='Cider Score Loss')
ax2.set_ylabel('Cider Score Loss')


# Rotating X-axis labels
ax.tick_params(axis='x', rotation=90)

ax.legend(loc="lower right")
ax2.legend(loc="lower left")


plt.title(BASE_DIR.split("/")[-1]+ ".bs."+str(BATCH_SIZE)+' ToMe Breakdown (ms) & Cider Score Drop')
plt.grid()

# Show plot
fig.show()
plt.savefig("./graph.pdf")

# %%

