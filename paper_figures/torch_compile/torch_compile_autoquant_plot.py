# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 

n_gpu=1
ns=0

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
            timer_result[k].sort(reverse=True)
    else:
        print("File doesn't exist: " + file_path)
    return timer_result

def get_folder(dataset, bs, exp_name):
    if "torch_compile" in exp_name:
        chameleon_prefix = "compile_test"
    else:
        chameleon_prefix = "cm3v21_30b_test"

    if dataset == "MSCOCO":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Flickr30k":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "TextVQA":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "OKVQA":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Vizwiz":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
    elif dataset == "Hellaswag":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    elif dataset == "Arc_easy":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_txt/"+chameleon_prefix+".mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
    elif dataset == "HumanEval":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-34b-hf/batch_size_"+str(bs)
    elif dataset == "HumanEval_small":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HumanEval_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    elif dataset == "MBPP_small":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/MBPP_codellama/meta-llama/CodeLlama-7b-hf/batch_size_"+str(bs)
    elif dataset == "Coco_Image":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "Partiprompts":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
    elif dataset == "S2ST":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/S2ST/batch_size_"+str(bs)+"/"
    elif dataset == "S2TT":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/S2TT/batch_size_"+str(bs)+"/"
    elif dataset == "T2TT":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/T2TT/batch_size_"+str(bs)+"/"
    elif dataset == "T2ST":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/T2ST/batch_size_"+str(bs)+"/"
    elif dataset == "HSTU-1M":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HSTU/num_embeddings_1000000_batch_size_"+str(bs)+"/"
    elif dataset == "HSTU-15M":
        return "/fsx-atom/yejinlee/paper_submission_results/"+exp_name+"/"+str(n_gpu)+"gpu_1node/HSTU/num_embeddings_15000000_batch_size_"+str(bs)+"/"
    else:
        assert False


baseline = [
    sum([np.average(v[5:]) for v in get_latency(get_folder("MSCOCO", 8, "latency_distribution_w_warmup")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("Vizwiz", 8, "latency_distribution_w_warmup")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("Coco_Image", 8, "latency_distribution_w_warmup")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("HumanEval", 1, "latency_distribution_w_warmup")).values()]),
    # sum([np.average(v[5:]) for v in get_latency(get_folder("HumanEval_small", 4, "latency_distribution_w_warmup")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("S2ST", 64, "torch_compile_baseline")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("S2TT", 64, "torch_compile_baseline")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("T2ST", 64, "torch_compile_baseline")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("T2TT", 64, "torch_compile_baseline")).values()]),
]


torch_compile = [
    sum([np.average(v[5:]) for v in get_latency(get_folder("MSCOCO", 8, "torch_compile")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("Vizwiz", 8, "torch_compile")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("Coco_Image", 8, "torch_compile")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("HumanEval", 1, "torch_compile")).values()]),
    # sum([np.average(v[5:]) for v in get_latency(get_folder("HumanEval_small", 4, "torch_compile")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("S2ST", 64, "torch_compile")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("S2TT", 64, "torch_compile")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("T2ST", 64, "torch_compile")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("T2TT", 64, "torch_compile")).values()]),
]


torch_compile_autoquant = [
    sum([np.average(v[5:]) for v in get_latency(get_folder("MSCOCO", 8, "torch_compile_autoquant")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("Vizwiz", 8, "torch_compile_autoquant")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("Coco_Image", 8, "torch_compile_autoquant")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("HumanEval", 1, "torch_compile_autoquant")).values()]),
    # sum([np.average(v[5:]) for v in get_latency(get_folder("HumanEval_small", 4, "torch_compile_autoquant")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("S2ST", 64, "torch_compile_autoquant")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("S2TT", 64, "torch_compile_autoquant")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("T2ST", 64, "torch_compile_autoquant")).values()]),
    sum([np.average(v[5:]) for v in get_latency(get_folder("T2TT", 64, "torch_compile_autoquant")).values()]),
]
  
# create data 
x = np.arange(len(baseline)) 
width = 0.2
  
# plot data in grouped manner of bar type 
plt.bar(x-0.2, baseline, width, color='cyan') 
plt.bar(x, torch_compile, width, color='orange') 
plt.bar(x+0.2, torch_compile_autoquant, width, color='green') 

plt.xticks(x, ['MSCOCO', 
                'Vizwiz',
                'Coco_Image',
               'HumanEval', 
            #    'HumanEval_small', 
               'S2ST', 
               'S2TT', 
               'T2ST', 
               'T2TT'], rotation=90) 
plt.xlabel("Teams") 
plt.ylabel("Average End-to-end Inference Tie (ms)") 
plt.legend(["Baseline", "Torch.compile", "Torch.compile+Autoquant"])
plt.grid(lw=0.2)
plt.show() 

# %%
