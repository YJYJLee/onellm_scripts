# %%
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import math

def get_timer_result_folder(dataset):
    if dataset == "MSCOCO":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    elif dataset == "Coco_Image":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "HumanEval":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/HumanEval_codellama/batch_size_1/"
    elif dataset == "MBPP":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/MBPP_codellama/batch_size_1/"
    elif dataset == "S2ST":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/S2ST/batch_size_1/"
    elif dataset == "S2TT":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/S2TT/batch_size_1/"
    elif dataset == "T2ST":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/T2ST/batch_size_1/"
    elif dataset == "T2TT":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/T2TT/batch_size_1/"
    elif dataset == "HSTU":
        return "/fsx-atom/yejinlee/paper_submission_results/latency_distribution_w_warmup/1gpu_1node/HSTU/num_embeddings_1000000_batch_size_1/"
    else:
        assert False

    # if dataset == "MSCOCO":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    # elif dataset == "Flickr30k":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    # elif dataset == "TextVQA":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    # elif dataset == "OKVQA":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    # elif dataset == "Vizwiz":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
    # elif dataset == "Coco_Image":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    # elif dataset == "Partiprompts":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    # elif dataset == "HumanEval":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/HumanEval_codellama/batch_size_1/"
    # elif dataset == "MBPP":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/MBPP_codellama/batch_size_1/"
    # elif dataset == "S2ST":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/S2ST/batch_size_1/"
    # elif dataset == "S2TT":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/S2TT/batch_size_1/"
    # elif dataset == "T2ST":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/T2ST/batch_size_1/"
    # elif dataset == "T2TT":
    #     return "/fsx-atom/yejinlee/sweep_final/1gpu_1node/T2TT/batch_size_1/"
    # else:
    #     assert False


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
        exit(0)



def plot_latency_histogram(dataset):
    print("DATASET: ", dataset)
    latencies = get_latency(get_timer_result_folder(dataset))#[15:]
    # latencies.sort()
    print(latencies)
    # latencies = latencies[:-25]
    print("LEN: ", len(latencies))

    # Get the index for 99th percentile
    p99_index = int(len(latencies) * 0.99)
    print("Index: ", p99_index, len(latencies))

    # Retrieve the latency at that index
    p99_latency = latencies[p99_index]

    min_latency = math.floor(min(latencies))
    max_latency = math.ceil(max(latencies))
    print("Min: ", min_latency, " Max: ", max_latency)

    bins = list()

    accum = int(min_latency)
    num_bin = 500
    interval=0
    while interval == 0:
        interval = int((max_latency-min_latency)/num_bin)
        num_bin /= 2
    num_bin = int(num_bin*2)
    print("num_bin: ", num_bin)
    for i in range(num_bin):
        bins.append(accum)
        accum+=interval

    fig = plt.figure(figsize=(8, 5))#, layout='tight')

    print("Plotting")
    N, binbin, patches = plt.hist(latencies, color='lightgreen', bins=bins, range=[min_latency, max_latency])
    # N, binbin, patches = plt.hist(latencies, bins=bins, range=[min_latency, max_latency])

    for bin_size, bin, patch in zip(N, binbin, patches):
        print(bin_size)
        if bin >= p99_latency:
            patch.set_facecolor("#FF0000")
        if bin == max(binbin[:-1]):
            patch.set_facecolor("#FF0000")
            patch.set_label("max")

        # if bin_size == max(N):
        #     patch.set_facecolor("#FF0000")
        #     patch.set_label("max")
        # elif bin_size == min(N):
        #     patch.set_facecolor("#00FF00")
        #     patch.set_label("min")

    print("AVG:", np.average(latencies))
    plt.legend()
    plt.title(dataset)
    plt.show()
    fig.savefig('/fsx-atom/yejinlee/analysis_figures/latency_distribution/'+dataset+'.pdf')


    # # Sort latencies in ascending order
    # latencies.sort()

    # # Get the index for 99th percentile
    # p99_index = int(len(latencies) * 0.99)

    # # Retrieve the latency at that index
    # p99_latency = latencies[p99_index]
    
# plot_latency_histogram("MSCOCO")
# plot_latency_histogram("Vizwiz")
# plot_latency_histogram("Coco_Image")
# plot_latency_histogram("HumanEval")
# plot_latency_histogram("MBPP")
# plot_latency_histogram("S2ST")
# plot_latency_histogram("S2TT")
# plot_latency_histogram("T2ST")
plot_latency_histogram("T2TT")
plot_latency_histogram("HSTU")
# %%
