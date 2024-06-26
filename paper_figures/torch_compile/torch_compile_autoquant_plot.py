# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 

n_gpu=1
ns=0

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
    return timer_result

def get_folder(dataset, bs, exp_name):
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

    if dataset == "MSCOCO-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    if dataset == "MSCOCO-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Coco_Image-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Coco_Image-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.coco_image."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_109m_sft.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_img/"+chameleon_prefix+".mn.cm3v21_109m_sft.t.partiprompts."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_109m_sft.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    # elif dataset == "Hellaswag":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    # elif dataset == "Arc_easy":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    elif dataset == "HumanEval-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "HumanEval-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP-34B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP-7B":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    # elif dataset == "S2ST":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/S2ST/batch_size_"+str(bs)+"/"
    # elif dataset == "S2TT":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/S2TT/batch_size_"+str(bs)+"/"
    # elif dataset == "T2TT":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/T2TT/batch_size_"+str(bs)+"/"
    # elif dataset == "T2ST":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/"+("batched" if exp_name=="latency_distribution_w_warmup" else "")+"/T2ST/batch_size_"+str(bs)+"/"
    # elif dataset == "HSTU-1M":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HSTU/num_embeddings_1000000_batch_size_"+str(bs)+"/"
    # elif dataset == "HSTU-15M":
    #     return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HSTU/num_embeddings_15000000_batch_size_"+str(bs)+"/"
    else:
        print(dataset)
        assert False


def plot(batch_size_dict):
    baseline = []

    for k, bs in batch_size_dict.items():
        baseline.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "latency_distribution_w_warmup")).values()]))

    # torch_compile_baseline = []
    # for k, bs in batch_size_dict.items():
    #     torch_compile_baseline.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "torch_compile_baseline")).values()]))

    # torch_compile_baseline = [a/b if b>0 else (2 if a>0 else 0) for a, b in zip(torch_compile_baseline, baseline)]

    torch_compile = []
    for k, bs in batch_size_dict.items():
        torch_compile.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "torch_compile")).values()]))

    torch_compile = [a/b if b>0 else (2 if a>0 else 0) for a, b in zip(torch_compile, baseline)]

    torch_compile_autoquant = []
    for k, bs in batch_size_dict.items():
        torch_compile_autoquant.append(sum([np.average(v) for v in get_latency(get_folder(k, bs, "torch_compile_autoquant")).values()]))
    torch_compile_autoquant = [a/b if b>0 else (2 if a>0 else 0) for a, b in zip(torch_compile_autoquant, baseline)]

    baseline = [b/b if b>0 else 0 for b in baseline]

    # create data 
    x = np.arange(len(baseline)) 
    width = 0.2

    fig, ax = plt.subplots(1, figsize=(10, 6), layout='tight')

    # plot data in grouped manner of bar type 
    # plt.bar(x-0.3, baseline, width, color='cyan') 
    # plt.bar(x-0.1, torch_compile_baseline, width, color='magenta') 
    # plt.bar(x+0.1, torch_compile, width, color='orange') 
    # plt.bar(x+0.3, torch_compile_autoquant, width, color='green') 

    plt.bar(x-0.2, baseline, width, color='cyan') 
    plt.bar(x, torch_compile, width, color='orange') 
    plt.bar(x+0.2, torch_compile_autoquant, width, color='green') 

    plt.xticks(x, list(batch_size_dict.keys()), rotation=90) 
    plt.xlabel("Workloads") 
    plt.ylabel("Normalized Speedup") 
    plt.title("Batch Size" + str(batch_size_dict["MSCOCO-34B"]))
    # plt.legend(["Baseline", "Torch.compile Baseline", "Torch.compile", "Torch.compile+Autoquant"],ncol=4)
    plt.legend(["Baseline", "Torch.compile", "Torch.compile+Autoquant"], ncol=3, bbox_to_anchor=(0.5, 1.15), loc="upper center")
    
    plt.grid(lw=0.2)
    plt.savefig("/fsx-atom/yejinlee/analysis_figures/torch_compile/batch_size_"+str(batch_size_dict["MSCOCO-34B"])+".pdf", bbox_inches = 'tight')
    plt.show() 


batch_size_dict = {
    "MSCOCO-34B": 1,
    "MSCOCO-7B": 1,
    "Flickr30k-34B": 1,
    "Flickr30k-7B": 1,
    "TextVQA-34B": 1,
    "TextVQA-7B": 1,
    "OKVQA-34B": 1,
    "OKVQA-7B": 1,
    "Vizwiz-34B": 1,
    "Vizwiz-7B": 1,
    "Coco_Image-34B": 1,
    "Coco_Image-7B": 1,
    "Partiprompts-34B": 1,
    "Partiprompts-7B": 1,
    "HumanEval-34B": 1,
    "HumanEval-7B": 1,
    "MBPP-34B": 1,
    "MBPP-7B": 1,
    # "S2ST": 1,
    # "S2TT": 1,
    # "T2ST": 1,
    # "T2TT": 1,
}
batch_sizes = [1,4,8,16,32,64,128]
# batch_sizes = [1]
for bs in batch_sizes:
    for k, v in batch_size_dict.items():
        batch_size_dict[k] = bs

    plot(batch_size_dict)
# %%
