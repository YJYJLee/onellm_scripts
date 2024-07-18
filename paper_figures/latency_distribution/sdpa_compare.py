# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
from statistics import geometric_mean
from matplotlib import colormaps

n_gpu=1
ns=0

# DIR_PREFIX=""
# DIR_PREFIX="./onellm_scripts/data_for_paper/latency_dist/"
DIR_PREFIX="/Users/yejinlee/hpca_2025/onellm_scripts/data_for_paper/latency_dist/"

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

def get_folder(dataset, bs, exp_name, sdpa=True):
    if "torch_compile" == exp_name:
        chameleon_prefix = "compile_test"
    elif "torch_compile_baseline" in exp_name:
        chameleon_prefix = "compile_test_baseline"
    elif "torch_compile_autoquant" in exp_name:
        chameleon_prefix = "quant_compile"
    else:
        if "7B" in dataset:
            chameleon_prefix = "cm3v21_109m_sft_test"
        else:
            chameleon_prefix = "cm3v21_30b_test"


    sdpa_str = "wo_sdpa/" if sdpa == False else ""
    if dataset == "MSCOCO\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    if dataset == "MSCOCO\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Coco_Image\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Coco_Image\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.coco_image."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_109m_sft.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.partiprompts."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_109m_sft.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    # elif dataset == "Hellaswag":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    # elif dataset == "Arc_easy":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    elif dataset == "HumanEval\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "HumanEval\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP\n34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP\n7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    elif dataset == "S2ST":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node//S2ST/batch_size_"+str(bs)+"/"
    elif dataset == "S2TT":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node//S2TT/batch_size_"+str(bs)+"/"
    elif dataset == "T2TT":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node//T2TT/batch_size_"+str(bs)+"/"
    elif dataset == "T2ST":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node//T2ST/batch_size_"+str(bs)+"/"
    # elif dataset == "HSTU-1M":
    #     return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/HSTU/num_embeddings_1000000_batch_size_"+str(bs)+"/"
    # elif dataset == "HSTU-15M":
    #     return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/HSTU/num_embeddings_15000000_batch_size_"+str(bs)+"/"
    else:
        print(dataset)
        assert False


def plot(batch_size_dict):
    baseline = []
    for k, bs in batch_size_dict.items():
        baseline.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "latency_distribution_w_warmup", sdpa=False)).values()]))

    baseline_w_sdpa = []
    for k, bs in batch_size_dict.items():
        baseline_w_sdpa.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "latency_distribution_w_warmup")).values()]))

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


    plt.xticks(x, list(batch_size_dict.keys()))#, rotation=90)
    # plt.ylim(0, 2) 
    plt.xlabel("Workloads") 
    plt.ylabel("Normalized Speedup") 
    # plt.title("Batch Size" + str(batch_size_dict["MSCOCO-34B"]))
    # plt.legend(["w/o SDPA", "w/ SDPA"], ncol=2, bbox_to_anchor=(0.5, 1.15), loc="upper center")
    
    plt.grid(lw=0.2)
    dump_dir = './onellm_scripts/analysis_figures/latency_distribution/sdpa/'
    os.makedirs(dump_dir, exist_ok=True)
    plt.savefig(dump_dir+"/batch_size_"+(str(batch_size_dict["MSCOCO\n34B"]) if len(set(batch_size_dict.values()))==1 else "max")+".pdf", bbox_inches = 'tight')
    print("Saving to ", dump_dir+"/batch_size_"+(str(batch_size_dict["MSCOCO\n34B"]) if len(set(batch_size_dict.values()))==1 else "max")+".pdf")
    plt.show() 


# batch_size_dict = {
#     "MSCOCO-34B": 1,
#     # "MSCOCO-7B": 1,
#     "Flickr30k-34B": 1,
#     # "Flickr30k-7B": 1,
#     "TextVQA-34B": 1,
#     # "TextVQA-7B": 1,
#     "OKVQA-34B": 1,
#     # "OKVQA-7B": 1,
#     "Vizwiz-34B": 1,
#     # "Vizwiz-7B": 1,
#     "Coco_Image-34B": 1,
#     # "Coco_Image-7B": 1,
#     "Partiprompts-34B": 1,
#     # "Partiprompts-7B": 1,
#     "HumanEval-34B": 1,
#     # "HumanEval-7B": 1,
#     "MBPP-34B": 1,
#     # "MBPP-7B": 1,
#     "S2ST": 1,
#     "S2TT": 1,
#     "T2ST": 1,
#     "T2TT": 1,
# }
# batch_sizes = [1,4,8,16,32,64,128]
# # batch_sizes = [1]
# for bs in batch_sizes:
#     for k, v in batch_size_dict.items():
#         batch_size_dict[k] = bs

#     plot(batch_size_dict)


batch_size_dict = {
    "MSCOCO\n34B": 16,
    # "MSCOCO-7B": 1,
    # "Flickr30k-34B": 1,
    # "Flickr30k-7B": 1,
    # "TextVQA-34B": 1,
    # "TextVQA-7B": 1,
    # "OKVQA-34B": 1,
    # "OKVQA-7B": 1,
    "Vizwiz\n34B": 16,
    # "Vizwiz-7B": 1,
    "Coco_Image\n34B": 16,
    # "Coco_Image-7B": 1,
    # "Partiprompts-34B": 1,
    # "Partiprompts-7B": 1,
    "HumanEval\n34B": 4,
    # "HumanEval-7B": 1,
    # "MBPP-34B": 1,
    # "MBPP-7B": 1,
    "S2ST": 128,
    "S2TT": 128,
    "T2ST": 384,
    "T2TT": 384,
}
plot(batch_size_dict)

batch_size_dict = {
    "MSCOCO\n34B": 1,
    # "MSCOCO-7B": 1,
    # "Flickr30k-34B": 1,
    # "Flickr30k-7B": 1,
    # "TextVQA-34B": 1,
    # "TextVQA-7B": 1,
    # "OKVQA-34B": 1,
    # "OKVQA-7B": 1,
    "Vizwiz\n34B": 1,
    # "Vizwiz-7B": 1,
    "Coco_Image\n34B": 1,
    # "Coco_Image-7B": 1,
    # "Partiprompts-34B": 1,
    # "Partiprompts-7B": 1,
    "HumanEval\n34B": 1,
    # "HumanEval-7B": 1,
    # "MBPP-34B": 1,
    # "MBPP-7B": 1,
    "S2ST": 1,
    "S2TT": 1,
    "T2ST": 1,
    "T2TT": 1,
}
plot(batch_size_dict)
# %%
