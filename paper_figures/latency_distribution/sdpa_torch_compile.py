# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import colormaps
from statistics import geometric_mean

n_gpu=1
ns=0

# DIR_PREFIX="./onellm_scripts/data_for_paper/compile_graph/"
# DIR_PREFIX="/data/home/yejinlee/RAG/onellm-eval/onellm_scripts/data_for_paper/compile_graph/"


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
        for k, v in timer_result.items():
            # timer_result[k].sort(reverse=True)
            timer_result[k] = reject_outliers(v)
        # if "vizwiz" in file_path and "compile_test" in file_path:
        #     print(timer_result["Generation"])
        #     exit(0)
    else:
        print("File doesn't exist: " + file_path)
        timer_result={"Dummy": 0}
    return timer_result

def get_folder(dataset, bs, exp_name, sdpa=True):
    if "torch_compile" == exp_name:
        chameleon_prefix = "compile_test"
    # elif "torch_compile_baseline" in exp_name:
    #     chameleon_prefix = "compile_test_baseline"
    elif "torch_compile_autoquant" in exp_name:
        chameleon_prefix = "quant_compile"
    else:
        if "7B" in dataset:
            chameleon_prefix = "cm3v21_109m_sft_test"
        else:
            chameleon_prefix = "cm3v21_30b_test"

    if "latency_distribution_w_warmup" in exp_name:
        DIR_PREFIX="/data/home/yejinlee/RAG/onellm-eval/onellm_scripts/data_for_paper/latency_dist/"
    else:
        DIR_PREFIX="/data/home/yejinlee/RAG/onellm-eval/onellm_scripts/data_for_paper/compile_graph/"

    sdpa_str = "wo_sdpa/" if sdpa == False else ""
    if dataset == "MSCOCO-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    if dataset == "MSCOCO-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Coco_Image-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Coco_Image-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.coco_image."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_109m_sft.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.partiprompts."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_109m_sft.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    # elif dataset == "Hellaswag":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    # elif dataset == "Arc_easy":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    elif dataset == "HumanEval-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "HumanEval-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP-34B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP-7B":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    elif dataset == "S2ST":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/S2ST/batch_size_"+str(bs)+"/"
    elif dataset == "S2TT":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/S2TT/batch_size_"+str(bs)+"/"
    elif dataset == "T2TT":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/T2TT/batch_size_"+str(bs)+"/"
    elif dataset == "T2ST":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+sdpa_str+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/T2ST/batch_size_"+str(bs)+"/"
    elif dataset == "HSTU-Pytorch":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/latency_distribution/pytorch/batch_size_"+str(bs)+"/"
    elif dataset == "HSTU-Triton":
        return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/latency_distribution/batch_size_"+str(bs)+"/"
    else:
        print(dataset)
        assert False


def plot(batch_size_dict, first_axis, second_axis, third_axis):
    baseline_wo_sdpa = []
    for k, bs in batch_size_dict.items():
        if k=="HSTU":
            baseline_wo_sdpa.append(sum([np.average(v) for v in get_latency(get_folder("HSTU-Pytorch", bs, "latency_distribution_w_warmup", sdpa=False)).values()]))
        else:
            baseline_wo_sdpa.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "latency_distribution_w_warmup", sdpa=False)).values()]))

    sdpa = []
    for k, bs in batch_size_dict.items():
        if k=="HSTU":
            sdpa.append(sum([np.average(v) for v in get_latency(get_folder("HSTU-Triton", bs, "latency_distribution_w_warmup")).values()]))
        else:
            sdpa.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "latency_distribution_w_warmup")).values()]))
    sdpa = [a/b if b>0 else (2 if a>0 else 0) for a, b in zip(baseline_wo_sdpa, sdpa)]


    torch_compile = []
    for k, bs in batch_size_dict.items():
        if k=="HSTU":
            torch_compile.append(0)
        else:
            torch_compile.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "torch_compile")).values()]))

    torch_compile = [b/a if a>0 else (2 if a>0 else 0) for a, b in zip(torch_compile, baseline_wo_sdpa)]

    torch_compile_autoquant = []
    for k, bs in batch_size_dict.items():
        if k=="HSTU":
            torch_compile_autoquant.append(0)
        else:
            torch_compile_autoquant.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "torch_compile_autoquant")).values()]))
    torch_compile_autoquant = [b/a if a>0 else (2 if a>0 else 0) for a, b in zip(torch_compile_autoquant, baseline_wo_sdpa)]


    # create data 
    # sdpa = [2,2,2,2,2,2,2,2,2,2,2,2,2]
    # torch_compile = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    num_llama_chameleon = 4
    # x = np.arange(len(baseline_wo_sdpa)) 
    x = np.arange(len(baseline_wo_sdpa)-num_llama_chameleon)*1.5
    # x = np.concatenate(np.arange(num_llama_chameleon), np.array([i*0.5 for i in range(len(baseline_wo_sdpa)-num_llama_chameleon)]))
    width = 0.2
    fig, ax = plt.subplots(1, figsize=(10, 6), layout='tight')

    # plot data in grouped manner of bar type 
    # plt.bar(x[:num_llama_chameleon]-0.35, sdpa[:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[6]) 
    # plt.bar(x[:num_llama_chameleon]-0.15, torch_compile[:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[4]) 
    # plt.bar(x[num_llama_chameleon:-1]-0.1, sdpa[num_llama_chameleon*2:-1], width, color=colormaps['Set3'].colors[6]) 
    # # HSTU
    # plt.bar(x[-1:], sdpa[-1:], width, color=colormaps['Set3'].colors[6]) 

    # plt.bar(x[:num_llama_chameleon]+0.15, sdpa[1:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[6]) 
    # plt.bar(x[:num_llama_chameleon]+0.35, torch_compile[1:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[4]) 
    # plt.bar(x[num_llama_chameleon:-1]+0.1, torch_compile[num_llama_chameleon*2:-1], width, color=colormaps['Set3'].colors[4]) 

    plt.bar(x[:num_llama_chameleon]-0.55, sdpa[:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[6]) 
    plt.bar(x[:num_llama_chameleon]-0.35, torch_compile[:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[4]) 
    plt.bar(x[:num_llama_chameleon]-0.15, torch_compile_autoquant[:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[2]) 
    plt.bar(x[num_llama_chameleon:-1]-0.1, sdpa[num_llama_chameleon*2:-1], width, color=colormaps['Set3'].colors[6]) 
    # HSTU
    plt.bar(x[-1:], sdpa[-1:], width, color=colormaps['Set3'].colors[6]) 

    plt.bar(x[:num_llama_chameleon]+0.15, sdpa[1:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[6]) 
    plt.bar(x[:num_llama_chameleon]+0.35, torch_compile[1:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[4]) 
    plt.bar(x[:num_llama_chameleon]+0.55, torch_compile_autoquant[1:num_llama_chameleon*2:2], width, color=colormaps['Set3'].colors[2]) 
    plt.bar(x[num_llama_chameleon:-1]+0.1, torch_compile[num_llama_chameleon*2:-1], width, color=colormaps['Set3'].colors[4]) 



    # plt.bar(x-0.2, baseline, width, color='cyan') 
    # plt.bar(x-0.1, sdpa, width, color=colormaps['Set3'].colors[6]) 
    # plt.bar(x+0.1, torch_compile, width, color=colormaps['Set3'].colors[4]) 


    # print("34B torch.compile speedup: ", geometric_mean([t for idx, t in enumerate(torch_compile) if idx%2==0 and t!=0]))
    # print("7B torch.compile speedup: ", geometric_mean([t for idx, t in enumerate(torch_compile) if idx%2==1 and t!=0]))

    # xxlabel = [val for pair in zip([xx-shift for xx in x[:num_separated_bar]], [xx+shift for xx in x[:num_separated_bar]]) for val in pair]+list(x[num_separated_bar:])

    # plt.xticks(list(np.arange(2*num_llama_chameleon)) + [x+2*num_llama_chameleon for x in np.arange((len(batch_size_dict)-num_llama_chameleon*2))], ["34B", "7B"]*num_llama_chameleon + [""]*(len(batch_size_dict)-num_llama_chameleon*2), fontsize=6)
    plt.xticks([val for pair in zip(x[:num_llama_chameleon]-0.35, x[:num_llama_chameleon]+0.35) for val in pair] + \
        list(x[num_llama_chameleon:]), \
        ["34B", "7B"]*num_llama_chameleon + [""]*(len(batch_size_dict)-num_llama_chameleon*2))#, fontsize=6)

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks([0, 1.5, 3.0, 4.5, 6, 7.5, 9, 10.5, 12], labels=["\n"+xl for xl in first_axis])#, fontsize=6)

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks([0, 1.5, 3.0, 4.5, 8.25, 12], labels=["\n\n"+xl for xl in second_axis])#, fontsize=6)


    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks([0, 3.0, 8.25, 12], labels=["\n\n\n"+xl for xl in third_axis])#, fontsize=8)


    # sec = ax.secondary_xaxis(location=0)
    # sec.set_xticks([0, 2, 4, 6.5], labels=["\n\n"+xl for xl in secondary_xlabel], fontsize=6)
    # # sec.tick_params(bottom = False) 
    # # sec.set_xticklabels([])
    # sec.set(xlabel=None)
    # sec.tick_params(bottom=False)  # remove the ticks


    # sec2 = ax.secondary_xaxis(location=0)
    # sec2.set_xticks([-0.5, 0.5, 3.5, 4.5, 8.5], labels=[])
    # sec2.tick_params('x', length=25, width=0.8)
    # ax.set_xlim(-0.5, 8.5)


    sec2 = ax.secondary_xaxis(location=0)
    sec2.set_xticks([-0.75, 0.725, 5.25, 11.25, 12.75], labels=[])
    sec2.tick_params('x', length=60, width=0.8)
    ax.set_xlim(-0.75, 12.75)


    ax.set_ylim(1, 10)

    # plt.xticks(x, list(batch_size_dict.keys()), rotation=90) 
    plt.xlabel("\n\n\nWorkloads", fontsize=14) 
    plt.ylabel("Normalized Speedup", fontsize=14) 
    # plt.title("Batch Size " +(str(batch_size_dict["MSCOCO-34B"]) if len(set(batch_size_dict.values()))==1 else "max"))
    # plt.legend(["Baseline", "Torch.compile Baseline", "Torch.compile", "Torch.compile+Autoquant"],ncol=4)
    # plt.legend(["Baseline", "Torch.compile", "Torch.compile+Autoquant"], ncol=3, bbox_to_anchor=(0.5, 1.15), loc="upper center")
    # plt.legend(["SDPA", "SDPA+Torch.compile"], ncol=2, bbox_to_anchor=(0.5, 1.15), loc="upper center")
    plt.legend(["SDPA", "SDPA+Torch.compile", "SDPA+Torch.compile+AutoQuant"], ncol=3, loc="upper center")
    
    plt.grid(lw=0.2)

    dump_dir = './onellm_scripts/analysis_figures/torch_compile/'
    os.makedirs(dump_dir, exist_ok=True)

    plt.savefig(dump_dir+"batch_size_"+(str(batch_size_dict["MSCOCO-34B"]) if len(set(batch_size_dict.values()))==1 else "max")+".pdf", bbox_inches = 'tight')
    print("Saving to ", dump_dir+"batch_size_"+(str(batch_size_dict["MSCOCO-34B"]) if len(set(batch_size_dict.values()))==1 else "max")+".pdf")
    plt.show() 


# batch_size_dict = {
#     "MSCOCO-34B": 1,
#     "MSCOCO-7B": 1,
#     "Flickr30k-34B": 1,
#     "Flickr30k-7B": 1,
#     "TextVQA-34B": 1,
#     "TextVQA-7B": 1,
#     "OKVQA-34B": 1,
#     "OKVQA-7B": 1,
#     "Vizwiz-34B": 1,
#     "Vizwiz-7B": 1,
#     "Coco_Image-34B": 1,
#     "Coco_Image-7B": 1,
#     "Partiprompts-34B": 1,
#     "Partiprompts-7B": 1,
#     "HumanEval-34B": 1,
#     "HumanEval-7B": 1,
#     "MBPP-34B": 1,
#     "MBPP-7B": 1,
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


# batch_size_dict = {
#     "HumanEval-34B": 4,
#     "HumanEval-7B": 4,
#     "MSCOCO-34B": 8,
#     "MSCOCO-7B": 64,
#     "Vizwiz-34B": 8,
#     "Vizwiz-7B": 32,
#     "Coco_Image-34B": 16,
#     "Coco_Image-7B": 16,
#     "S2ST": 64,
#     "S2TT": 64,
#     "T2ST": 64,
#     "T2TT": 64,
# }
# first_axis = [
#     "HumanEval",
#     "MSCOCO",
#     "Vizwiz",
#     "Coco Image",
#     "Fleurs"
# ]
# second_axis = [
#     "T-T",
#     "Chameleon",
#     "Seamless",
# ]
# third_axis = [
#     "Llama",
#     "Chameleon",
#     "Seamless",
# ]

# plot(batch_size_dict, first_axis, second_axis, third_axis)

batch_size_dict = {
    "HumanEval-34B": 1,
    "HumanEval-7B": 1,
    "MSCOCO-34B": 1,
    "MSCOCO-7B": 1,
    "Vizwiz-34B": 1,
    "Vizwiz-7B": 1,
    "Coco_Image-34B": 1,
    "Coco_Image-7B": 1,
    "S2ST": 1,
    "S2TT": 1,
    "T2ST": 1,
    "T2TT": 1,
    "HSTU": 1,
}

first_axis = [
    "T-T",
    "I-T",
    "IT-T",
    "T-I",
    "S-S",
    "S-T",
    "T-S",
    "T-T",
    "H-A",
]
second_axis = [
    "HumanEval",
    "MSCOCO",
    "Vizwiz",
    "Coco Image",
    "Fleurs",
    "Synthetic"
]
third_axis = [
    "Llama",
    "Chameleon",
    "Seamless",
    "HSTU"
]

plot(batch_size_dict, first_axis, second_axis, third_axis)

# %%
