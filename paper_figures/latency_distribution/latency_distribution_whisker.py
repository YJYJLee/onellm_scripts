# %%
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import colormaps


# DIR_PREFIX=""
DIR_PREFIX="/data/home/yejinlee/RAG/onellm-eval/onellm_scripts/data_for_paper/latency_dist/"

def get_timer_result_folder(dataset):
    if dataset == "MSCOCO":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "Coco_Image":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts.0_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "HumanEval":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-34b-hf/batch_size_1/"
    elif dataset == "MBPP":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/MBPP_codellama/meta-llama/CodeLlama-34b-hf/batch_size_1/"
    elif dataset == "S2ST":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/S2ST/batch_size_1/"
    elif dataset == "S2TT":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/S2TT/batch_size_1/"
    elif dataset == "T2ST":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/T2ST/batch_size_1/"
    elif dataset == "T2TT":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/T2TT/batch_size_1/"
    elif dataset == "HSTU-Pytorch":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/latency_distribution/pytorch/batch_size_1/"
    elif dataset == "HSTU-Triton":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/latency_distribution/batch_size_1/"
    else:
        assert False

def get_latency(file_path):
    file_path += "/timer_result.txt"
    timer_result = dict()
    if os.path.isfile(file_path):
        f = open(file_path, "r")
        print("Reading from ", file_path)
        headers = None
        num_line = 0
        for idx, sl in enumerate(f):
            if idx==0:
                headers = re.sub("\n", "", sl).split("\t")
                for h in headers:
                    timer_result[h] = list()
            else:
                slsl = [float(s) for s in re.sub("\n", "", sl).split("\t") if s!=""]
                for idx, h in enumerate(headers):
                    timer_result[h].append(slsl[idx])
                num_line+=1
        final_timer_result = list()
        for i in range(num_line):
            final_timer_result.append(sum([v[i] for v in timer_result.values()]))
            # print(sum([v[i] for v in timer_result.values()]))
            # exit(0)
        return final_timer_result

    else:
        print("File doesn't exist: " + file_path)
        return None


colors = colormaps['Set3'].colors

def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return [a for i, a in enumerate(data) if s[i] < m ]


def get_latency_dist(dataset):
    latencies = get_latency(get_timer_result_folder(dataset))#[15:]

    # latencies.sort()
    # print(latencies)
    # latencies = latencies[:-25]
    # print("LEN: ", len(latencies))
    latencies = reject_outliers(latencies)
    print("DATASET: ", dataset, " avg: ", np.average(latencies), " stdev: ", np.std(latencies))
    return latencies

data = list()
data.append(get_latency_dist("HSTU-Pytorch"))

data.append(get_latency_dist("S2ST"))
data.append(get_latency_dist("S2TT"))
data.append(get_latency_dist("T2ST"))
data.append(get_latency_dist("T2TT"))
data.append(get_latency_dist("MSCOCO"))
# data.append(get_latency_dist("Flickr30k"))
# data.append(get_latency_dist("TextVQA"))
# data.append(get_latency_dist("OKVQA"))
data.append(get_latency_dist("Vizwiz"))
data.append(get_latency_dist("Coco_Image"))
# data.append(get_latency_dist("Partiprompts"))
data.append(get_latency_dist("HumanEval"))
data.append(get_latency_dist("MBPP"))

plt.figure(figsize=(64, 3))

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, facecolor='w',
                                  gridspec_kw={'width_ratios': (5, 40, 5)})

bp = list()
bp.append(ax1.boxplot(data, patch_artist = True,
                notch ='True', vert = 0))

bp.append(ax2.boxplot(data, patch_artist = True,
                notch ='True', vert = 0))
bp.append(ax3.boxplot(data, patch_artist = True,
                notch ='True', vert = 0))
 


ax1.set_xlim(45,47)  # x-axis range limited to 0 - 100 
ax2.set_xlim(100,7000)  # x-axis range limited to 0 - 100 
ax3.set_xlim(107900, 108400)  # x-axis range limited to 250 - 300


# hide the spines between ax and ax2
# ax1.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax1.yaxis.tick_left()
# ax.tick_params(labelright='off')

ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax2.tick_params(axis='y', length=0)
ax3.tick_params(axis='y', length=0)


# Draw the diagonal lines to show broken axes
d = 2.  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
ax2.plot([1, 1], [0, 1], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 0], [0, 1], transform=ax3.transAxes, **kwargs)


# colors = ['#0000FF', '#00FF00', 
#           '#FFFF00', '#FF00FF']

for i in range(len(bp)):
    for patch, color in zip(bp[i]['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
    
    # changing color and linewidth of
    # whiskers
    for whisker in bp[i]['whiskers']:
        # whisker.set(color ='#8B008B',
        whisker.set(color ='grey',
                    linewidth = 1.5,
                    linestyle =":")
    
    # changing color and linewidth of
    # caps
    for cap in bp[i]['caps']:
        # cap.set(color ='#8B008B',
        cap.set(color ='purple',
                linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp[i]['medians']:
        median.set(color ='red',
                linewidth = 1)
    
    # changing style of fliers
    for flier in bp[i]['fliers']:
        flier.set(marker ='D',
        # flier.set(marker ='x',
                color ='#e7298a',
                alpha = 0.5)
        

ax1.set_yticklabels([
"[H-A] Synthetic",
"[S-S] Fleurs",
"[S-T] Fleurs",
"[T-S] Fleurs",
"[T-T] Fleurs",
"[I-T] MSCOCO",
# "Flickr30k",
# "TextVQA",
# "OKVQA",
"[IT-T] Vizwiz",
"[T-I] Coco_Image",
# "Partiprompts",
"[T-T] HumanEval",
"[T-T] MBPP",
]*3)



sec = ax1.secondary_yaxis(location=0)
sec.set_yticks([9.5, 7, 3.5, 1], labels=["Llama                          ", 
                                         "Chameleon                      ", 
                                         "Seamless                       ",
                                         "HSTU                           "], fontsize=12)
# sec.set_yticklabels([])
# sec.get_yaxis().set_visible(False)


sec2 = ax1.secondary_yaxis(location=0)
# sec2.set_yticks([1.5, 5.5, 8.5], labels=[])
sec2.set_yticks([0.5, 1.5, 5.5, 8.5, 10.5], labels=[])
sec2.tick_params('y', length=160, width=0.8)
ax1.set_ylim(0.5, 10.5)

# f.xlabel("Inference Time (ms)", fontsize=10) 


# # Adding title 
# plt.title("Customized box plot")
 
# Removing top axes and right axes
# ticks
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

# show plot
plt.show()
dump_dir = '/data/home/yejinlee/RAG/onellm-eval/onellm_scripts/analysis_figures/latency_dist'
os.makedirs(dump_dir, exist_ok=True)
f.savefig(dump_dir+'/whisker.pdf', bbox_inches = 'tight')
print("Saving to "+ dump_dir+'/whisker.pdf')