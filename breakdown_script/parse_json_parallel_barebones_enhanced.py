'''
Usage: python torch_profiler_parser.py <path to directory of traces> <prefix>
Example: python torch_profiler_parser.py /private/home/yejinlee/profiler_results/profiler_trace_tb_torch2.1_SPEECH_TO_SPEECH_module_breakdown_8gpu_batch_size1 YJ_PROFILE
'''

# %%
import json
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock, Manager

import sys, os
import csv 
import argparse
import itertools

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib import colormaps
import re

# Initialize ArgParser:
parser = argparse.ArgumentParser("argparser")

# Parse possible arguments:
parser.add_argument("--json-file", type=str, default="profile.json")
parser.add_argument("--desired-prefixes", type=str, default="RESBLOCK_AG*ATTENTION_AG*FEEDFORWARD_AG")
parser.add_argument("--graph-path", type=str, default="/fsx-checkpoints/yejinlee/analysis_figures")
parser.add_argument("--batch-size", action="store_true", default=False)
parser.add_argument("--n-retrieved-doc", action="store_true", default=False)
parser.add_argument("--both", action="store_true", default=False)
parser.add_argument("--compare-efficient-attn", action="store_true", default=False)
parser.add_argument("--compare-dir", type=str, default="")
parser.add_argument("--multigpu", action="store_true", default=False)


# Get arguments into readable format:
args = parser.parse_args()
print("Finish parsing arguments")
print("args = ", args)


prefix = "AG"

# desired_prefixes = ["RESBLOCK_AG", "ATTENTION_AG"]
# desired_prefixes = ['MODULE_Embedding_AG', 'MODULE_LayerNorm_AG', 'MODULE_Linear_AG', 'MODULE_QuickGELUActivation_AG', 'MODULE_Timesteps_AG', 'MODULE_SiLU_AG', 'MODULE_Conv2d_AG', 'MODULE_GroupNorm_AG', 'MODULE_Dropout_AG', 'MODULE_Attention_AG']
print("args.desired_prefixes= ", args.desired_prefixes)
desired_prefixes = args.desired_prefixes.split("*")

print("NOW DESIRED PREFIXES = ", desired_prefixes)
top_prefixes = ["DENOISING_LOOP"]


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
def parse_file(file_path, save_folder_path=None, plot_graph=True):
    kernel_breakdown = {}
    desired_prefixes_gpu_dur = []

    if os.path.isfile(file_path):
        file_p = open(file_path)
        print("Reading from ", file_path)
    else:
        print("File doesn't exists!!!! ", file_path)
        return kernel_breakdown

    print("PARSING NOW ")
    jsonString = json.load(file_p)["traceEvents"]

    slices = dict()
    total_train_time = 0

    cpu_desired_ops = []
    cpu_top_ops = []
    
    gpu_launch_kernels=[]

    gpu_kernels=[]

    nccl_kernels=[]

    
    with tqdm(total=len(jsonString), desc="Profiling...", position=0) as pbar:
        ####################### FIRST LOOP THROUGH JSON #################################
        for idx,l in enumerate(jsonString):
            found_desired_prefix = False

            # IF ITS A CPU KERNEL:
            if "cat" in l and (l["cat"]=="cpu_op" or l["cat"]=="user_annotation") and "ts" in l and "dur" in l:
                
                for p in desired_prefixes:
                    if p in l["name"]:
                        cpu_desired_ops.append(l)
                        found_desired_prefix = True

                if found_desired_prefix == False:
                    for top in top_prefixes:
                        if top in l["name"]:
                            cpu_top_ops.append(l)


            # IF ITS A GPU KERNEL:
            if "cat" in l and l["cat"]=="kernel":

                # If there is not a flow event right after this, error
                if jsonString[idx+1]["ph"] != "f":
                    print("ERROR")
                    exit(0)

                if "nccl" in l["name"]:
                    nccl_kernels.append(l)
                else:
                    found=False
                    # gpu_kernels.append(l) 
                    # gpu_launch_kernels.append(jsonString[idx+2]) 
                    
                    # assert jsonString[idx+2]["name"] == "cudaLaunchKernel"
                    # assert l["args"]["External id"] == jsonString[idx+2]["args"]["External id"]


                    for ii in range(idx+2,len(jsonString)): # should find the launch kernel closest to this event
                        if jsonString[ii]["name"] == "cudaLaunchKernel": # if there is a launch kernel there
                            gpu_launch_kernels.append(jsonString[ii]) # From(cpu)
                            gpu_kernels.append(l) # To(gpu kernel) # append this to the GPU kernel
                            found=True
                            break
                    if found==False:
                        print("ERROR2")
                        

    # make sure you have a gpu launch kernel associated with every gpu kernel
    assert len(gpu_kernels) == len(gpu_launch_kernels)

    ###########################################
    print("FINISHED CATEGORIZING THE KERNELS")
    ###########################################
    # print("cpu_desired_ops = ", cpu_desired_ops)
    # print("gpu_kernels = ", gpu_kernels[2])
    # print("gpu_kernels = ", gpu_launch_kernels[2])
    # print("LEN CPU DESIRED OPS = ", len(cpu_desired_ops))

    kernel_breakdowns = {}

    ###########################################
    print("---- LINKING CPU ANNOTATIONS TO GPU KERNELS -----")
    ###########################################

    current_cpu_desired_index = 0
    current_cpu_top_index = 0
    current_cpu_desired_index_changed=False

    print("Start sorting")

    gpu_launch_kernels = sorted(gpu_launch_kernels, key=lambda x: x['ts'])
    gpu_kernels = sorted(gpu_kernels, key=lambda x: x['ts'])
    nccl_kernels = sorted(nccl_kernels, key=lambda x: x['ts'])
    cpu_desired_ops = sorted(cpu_desired_ops, key=lambda x: x['ts'])

    print("Finished sorting")

    min_launch = gpu_kernels[0]
    max_launch = gpu_kernels[0]
    decoding_step_time = list()
    end_of_decoding_step = None
    entered_scoring = False
    decoding_step_start_time = [gpu_kernels[0]["ts"]]
    
    # for idx,l in enumerate(jsonString):
    for idx, launch in tqdm(enumerate(gpu_launch_kernels)):

        # print("==========================================================================")

        time_start = launch["ts"]
        time_end = launch["ts"] + launch["dur"]

        # print("time_start = ", time_start)
        # print("time_end = ", time_end)

        corresp_gpu_kernel = gpu_kernels[idx]
        gpu_time_start = corresp_gpu_kernel["ts"]
        gpu_time_end = corresp_gpu_kernel["ts"] + corresp_gpu_kernel["dur"]

        if gpu_time_end >= max_launch["ts"]:
            max_launch = corresp_gpu_kernel

        found_cpu = False

        while current_cpu_desired_index < len(cpu_desired_ops) and found_cpu == False:
            print(current_cpu_desired_index,  found_cpu == False)
            cpu_op = cpu_desired_ops[current_cpu_desired_index]

            cpu_start = cpu_op["ts"]
            cpu_end = cpu_op["ts"]+cpu_op["dur"]
            print("cpu_op: ", cpu_op)
            print(cpu_start, " ", cpu_end)
            print("time_start: ", time_start)
            # print("cpu_op = ", cpu_op["name"])
            # current_cpu_desired_index_changed = False


            if entered_scoring == False:
                end_of_decoding_step = gpu_kernels[idx-1]

            # In annotation
            if cpu_start <= time_start and cpu_end >= time_end:
                name = cpu_op["name"]
                
                if name in kernel_breakdown.keys():
                    kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"]/1000
                else:
                    kernel_breakdown[name] = [corresp_gpu_kernel["dur"]/1000]

                found_cpu = True
                if name == "MODULE_SCORING_AG":
                    # print("SCORING ", corresp_gpu_kernel)
                    entered_scoring = True

            # past end of annotation
            elif cpu_end <= time_start:
                # if current_cpu_desired_index_changed==True and cpu_op["name"] == "MODULE_SCORING_AG":
                if current_cpu_desired_index < len(cpu_desired_ops) and cpu_op["name"] == "MODULE_SCORING_AG":
                    # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"])
                    # print(min_launch, " ", gpu_kernels[idx-1])

                    # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"], " / ", end_of_decoding_step["ts"]+end_of_decoding_step["dur"]-min_launch["ts"])
                    # print(min_launch, " ", end_of_decoding_step, " / idx-1: ", idx-1)
                    decoding_step_time.append((end_of_decoding_step["ts"]+end_of_decoding_step["dur"]-min_launch["ts"])/1000)
                    end_of_decoding_step = None
                    min_launch = max_launch
                    entered_scoring = False

                    for k, v in kernel_breakdown.items():
                        v.append(0)
                    decoding_step_start_time.append(gpu_time_start)


                # if also before end of next annotation
                if current_cpu_desired_index == len(cpu_desired_ops)-1 or time_end <= cpu_desired_ops[current_cpu_desired_index+1]["ts"]:
                    name =  "Copy" if "copy" in corresp_gpu_kernel["name"] else "Misc"

                    if name in kernel_breakdown.keys():
                        kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"]/1000
                    else:
                        kernel_breakdown[name] = [corresp_gpu_kernel["dur"]/1000]

                    found_cpu = True
                current_cpu_desired_index = current_cpu_desired_index + 1

            elif time_end <= cpu_start:
                name = "Copy" if "copy" in corresp_gpu_kernel["name"] else "Misc"

                if name in kernel_breakdown.keys():
                    kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"]/1000
                else:
                    kernel_breakdown[name] = [corresp_gpu_kernel["dur"]/1000]

                found_cpu = True


        # For ones at the end that haven't been assigned but are past the cpu annotations
        if found_cpu == False:
            name = "Copy" if "copy" in corresp_gpu_kernel["name"] else "Misc"
            if name in kernel_breakdown.keys():
                kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"]/1000
            else:
                kernel_breakdown[name] = [corresp_gpu_kernel["dur"]/1000]

    if cpu_op["name"] == "MODULE_SCORING_AG":

        # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"])
        # print(min_launch, " ", corresp_gpu_kernel)

        # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"], " / ", end_of_decoding_step["ts"]+end_of_decoding_step["dur"]-min_launch["ts"])
        # print(min_launch, " ", end_of_decoding_step, " / idx-1: ", idx-1)

        decoding_step_time.append((end_of_decoding_step["ts"]+end_of_decoding_step["dur"]-min_launch["ts"])/1000)


    if len(nccl_kernels) > 0:
        communication_list = [0]
        idx = 1
        for nc in nccl_kernels:
            if idx == len(decoding_step_start_time) or decoding_step_start_time[idx] >= nc['ts']:
                communication_list[idx-1] += nc['dur']/1000
            else:
                idx+=1
                if idx < len(decoding_step_start_time):
                    assert decoding_step_start_time[idx] >= nc['ts']
                
                communication_list.append(nc['dur']/1000)
    else:
        communication_list = [0]*len(decoding_step_start_time)

    kernel_breakdown["Communication"] = communication_list

    print("combined_kernel_breakdown = ", kernel_breakdown)

    for key, value in kernel_breakdown.items(): 
        print(key, " = ", value)
    print("end")
    
    kernel_breakdown["Linear"] = [a+b for a,b in zip(kernel_breakdown["MODULE_ColumnParallelLinear_AG"], kernel_breakdown["MODULE_RowParallelLinear_AG"])]
    del kernel_breakdown["MODULE_ColumnParallelLinear_AG"]
    del kernel_breakdown["MODULE_RowParallelLinear_AG"]

    if "MODULE_ToMe_Merge_AG" in kernel_breakdown:
        kernel_breakdown["MODULE_ToMe_Merge_AG"] = [0]*(len(kernel_breakdown["Misc"])-len(kernel_breakdown["MODULE_ToMe_Merge_AG"])) + kernel_breakdown["MODULE_ToMe_Merge_AG"]
    if plot_graph:
        graph_gpu_kernel_breakdown(kernel_breakdown, save_folder_path)

    idle_list = list()
    for s in np.arange(len(kernel_breakdown["MODULE_SCORING_AG"])):
        idle_list.append(decoding_step_time[s]-sum([(v[s]) if k!="MODULE_SCORING_AG" else 0 for k, v in kernel_breakdown.items()]))
    kernel_breakdown["Idle"] = idle_list

    if plot_graph:
        graph_gpu_kernel_breakdown_idle(kernel_breakdown, save_folder_path)
    # graph_overall(kernel_breakdown, decoding_step_time)
    print("FINISHED")
    return kernel_breakdown
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

colormap = colormaps['Set3'].colors
colormap2 = colormaps['Paired'].colors

cmap = {"MODULE_ParallelEmbedding_AG": colormap[0],
    "Misc": colormap[1],
    "MODULE_FusedRMSNorm_AG": colormap[2],
    "Linear": colormap[3],
    "MODULE_LayerNorm_AG": colormap[4],
    "MODULE__InnerAttention_AG": colormap[5],
    "Copy": colormap[7],
    "MODULE_SCORING_AG": colormap[8],
    "Idle": colormap[9],
    "Communication": colormap[10],
    "MODULE_PREPROC_ENCODE_IMAGES_AG": colormap[11],
    "MODULE_POSTPROC_GENERATE_TEXT_AG": colormap2[0],
}
    
def prep_graph(nested=False):
    print("GRAPHING")

    figures_dir = '.'

    dpi = 320

    page_width = 48 if args.multigpu and args.batch_size else 12
    lr_margin = 0.75
    column_separation = 0.25

    fig_width = (page_width - 2*lr_margin - column_separation)/2
    dbl_fig_width = page_width - 2*lr_margin
    fig_height = 0.45 * fig_width if not nested else 0.8 * fig_width
    
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), dpi=dpi, layout='tight')    # List of labels
    return fig, ax, dpi

def wrapup_graph(plt, ax, exp_name, xlabel, yaxis_title, title, save_folder_path, filename, dpi, nested=False):
    plt.ylabel(yaxis_title, fontsize=6)
    plt.xlabel(("\n\n\n\n\n\n\n\n\n\n\n" if nested else "") + exp_name, fontsize=6)
    plt.title(title, fontsize=6)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    ax.tick_params(axis='x', rotation=90 if len(xlabel[-1])>8 else 0)

    plt.grid(lw=0.2)
    ax.set_axisbelow(True)
    plt.show()

    file_path = save_folder_path+re.sub("\n", "", exp_name)+filename
    print("graph_path = ", file_path)
    plt.savefig(file_path, dpi=dpi)

def graph_gpu_kernel_breakdown(kernel_breakdown, save_folder_path):
    fig, ax, dpi = prep_graph()

    labels = list(kernel_breakdown.keys())

    steps_len = len(kernel_breakdown["MODULE_SCORING_AG"])
    steps = np.arange(steps_len)
    bottom = [0]*steps_len

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(steps, v, bottom = bottom, label=k, color=cmap[k], width=0.8)
        bottom = np.add(bottom, v)

    ax.legend(fontsize=4)

    folder_name_split = args.json_file.split("/")
    title_name = folder_name_split[-2] + "\nOperator Breakdown\n" + ("Warmup with 5 examples, profile result for 6th inference sample" if "t2i" in args.json_file else "Warmup with 10 examples, profile result for 11th inference sample")
    plt.title(title_name, fontsize=6)
    plt.ylabel('Execution Time (ms)', fontsize=6)
    plt.xlabel('Decoding Step', fontsize=6)

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)

    if "retrieval" not in args.json_file:
        if "i2t" in folder_name_split[-2]:
            plt.ylim(0,2500)
        elif "it2t" in folder_name_split[-2]:
            plt.ylim(0,2500)
        elif "t2i" in folder_name_split[-2]:
            plt.ylim(0,1400)
    else:
        if "bs4" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0,1400)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0,1400)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0,1000)
        elif "bs16" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0,5500)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0,5500)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0,1000)

    plt.grid(lw=0.2)
    ax.set_axisbelow(True)
    plt.show()

    # folder_path = args.graph_path+"/"+folder_name_split[-2]
    os.makedirs(save_folder_path, exist_ok=True)
    file_path = save_folder_path+"decoding_step_operator_breakdown.pdf"
    print("graph_path = ", file_path)
    plt.savefig(file_path, dpi=dpi)

def graph_gpu_kernel_breakdown_idle(kernel_breakdown, save_folder_path):
    fig, ax, dpi = prep_graph()

    labels = list(kernel_breakdown.keys())

    steps_len = len(kernel_breakdown["MODULE_SCORING_AG"])
    steps = np.arange(steps_len)
    bottom = [0]*steps_len

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(steps, v, bottom = bottom, label=k, color=cmap[k], width=0.8)
        bottom = np.add(bottom, v)

    ax.legend(fontsize=4)

    folder_name_split = args.json_file.split("/")
    title_name = folder_name_split[-2] + "\nOperator Breakdown w/ GPU Idle(Inference)\n" + ("Warmup with 5 examples, profile result for 6th inference sample" if "t2i" in args.json_file else "Warmup with 10 examples, profile result for 11th inference sample")
    plt.title(title_name, fontsize=6)
    plt.ylabel('Execution Time (ms)', fontsize=6)
    plt.xlabel('Decoding Step', fontsize=6)

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)

    if "retrieval" not in args.json_file:
        if "i2t" in folder_name_split[-2]:
            plt.ylim(0,2500)
        elif "it2t" in folder_name_split[-2]:
            plt.ylim(0,2500)
        elif "t2i" in folder_name_split[-2]:
            plt.ylim(0,1400)
    else:
        if "bs4" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0,1400)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0,1400)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0,1000)
        elif "bs16" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0,5500)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0,5500)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0,1000)

    plt.grid(lw=0.2)
    ax.set_axisbelow(True)
    plt.show()

    # folder_path = args.graph_path+"/"+folder_name_split[-2]
    os.makedirs(save_folder_path, exist_ok=True)
    file_path = save_folder_path+"decoding_step_operator_breakdown_idle.pdf"
    print("graph_path = ", file_path)
    plt.savefig(file_path, dpi=dpi)

def graph_overall(kernel_breakdown, xlabel, exp_name, save_folder_path, nested=False, secondary_xlabel=None):
    fig, ax, dpi = prep_graph()

    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [0]*steps_len

    # total_time = sum([v for v in kernel_breakdown.values()])

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(xlabel, v, bottom = bottom, label=k, color=cmap[k], width=0.8)
        bottom = np.add(bottom, v)

    ax.legend(fontsize=4)

    # if nested:
    #     sec = ax.secondary_xaxis(location=0)
    #     sec.set_xticks(x, labels=secondary_xlabel, fontsize=6)

    folder_name_split = args.json_file.split("/")
    title_name = folder_name_split[-1] + " " + exp_name + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n" + ("Warmup with 5 examples, profile result for 6th inference sample" if "t2i" in args.json_file else "Warmup with 10 examples, profile result for 11th inference sample")

    wrapup_graph(plt, ax, exp_name, xlabel, 'Execution Time Breakdown (ms)', title_name, save_folder_path, "_decoding_step_operator_breakdown_overall.pdf", dpi, nested=nested)


def graph_overall_ratio(kernel_breakdown, xlabel, exp_name, save_folder_path, nested=False, secondary_xlabel=None):
    fig, ax, dpi = prep_graph()
    
    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [0]*steps_len

    total_time = list()
    for i in range(steps_len):
        total_time.append(sum([v[i] for v in kernel_breakdown.values()]))

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(xlabel, [vv/t*100 if t!=0 else 0 for vv, t in zip(v, total_time)], bottom = bottom, label=k, color=cmap[k], width=0.8)
        bottom = np.add(bottom, [vv/t*100 if t!=0 else 0 for vv, t in zip(v, total_time)])

    ax.legend(fontsize=4)

    # if nested:
    #     sec = ax.secondary_xaxis(location=0)
    #     sec.set_xticks(x, labels=secondary_xlabel, fontsize=6)

    folder_name_split = args.json_file.split("/")
    title_name = folder_name_split[-1] + " " + exp_name + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n" + ("Warmup with 5 examples, profile result for 6th inference sample" if "t2i" in args.json_file else "Warmup with 10 examples, profile result for 11th inference sample")

    wrapup_graph(plt, ax, exp_name, xlabel, 'Execution Time Breakdown (%)', title_name, save_folder_path, "_decoding_step_operator_breakdown_overall_ratio.pdf", dpi, nested=nested)


def graph_overall_compare(kernel_breakdown, compare_breakdown, xlabel, exp_name, save_folder_path):
    fig, ax, dpi = prep_graph(nested=True)

    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [0]*steps_len
    bottom_compare = [0]*steps_len
    x = np.arange(steps_len)

    shift=0.2

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        # ax.bar(xlabel, v, bottom = bottom, label=k, color=cmap[k], width=0.8)
        ax.bar([xx-shift for xx in x], v if len(kernel_breakdown) != 0 else [0]*len(x), bottom=bottom, label=k, color=cmap[k], width=0.35)
        ax.bar([xx+shift for xx in x], compare_breakdown[k] if len(compare_breakdown) != 0 else [0]*len(x), bottom=bottom_compare, color=cmap[k], width=0.35)
        bottom = np.add(bottom, v)
        bottom_compare = np.add(bottom_compare, compare_breakdown[k])

    ax.legend(fontsize=4)
    plt.xticks([val for pair in zip([xx-shift for xx in x], [xx+shift for xx in x]) for val in pair], ["w/", "w/o"]*steps_len, fontsize=6)

    sec = ax.secondary_xaxis(location=0)
    if len(xlabel) == 20:
        sec.set_xticks(x, labels=[xl+"        " for xl in xlabel], fontsize=6, rotation=90)
    else:
        sec.set_xticks(x, labels=["\n"+xl for xl in xlabel], fontsize=6)
    sec.tick_params(bottom = False) 

    folder_name_split = args.json_file.split("/")
    title_name = folder_name_split[-1] + " " + exp_name + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n" + ("Warmup with 5 examples, profile result for 6th inference sample" if "t2i" in args.json_file else "Warmup with 10 examples, profile result for 11th inference sample")
    wrapup_graph(plt, ax, exp_name, xlabel, 'Execution Time Breakdown (ms)', title_name, save_folder_path, "_decoding_step_operator_breakdown_overall.pdf", dpi, nested=True)

def graph_overall_ratio_compare(kernel_breakdown, compare_breakdown, xlabel, exp_name, save_folder_path):
    fig, ax, dpi = prep_graph(nested=True)
    assert kernel_breakdown.keys() == compare_breakdown.keys()
    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [0]*steps_len
    bottom_compare = [0]*steps_len
    x = np.arange(steps_len)

    shift=0.2

    total_time = list()
    total_time_compare = list()
    for i in range(steps_len):
        total_time.append(sum([v[i] for v in kernel_breakdown.values()]))
        total_time_compare.append(sum([v[i] for v in compare_breakdown.values()]))

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar([xx-shift for xx in x], [vv/t*100 if t!=0 else 0 for vv, t in zip(v, total_time)] if len(kernel_breakdown) != 0  else [0]*len(x), bottom = bottom, label=k, color=cmap[k], width=0.35)
        ax.bar([xx+shift for xx in x], [vv/t*100 if t!=0 else 0 for vv, t in zip(compare_breakdown[k], total_time_compare)] if len(compare_breakdown) != 0  else [0]*len(x), bottom = bottom_compare, color=cmap[k], width=0.35)
        bottom = np.add(bottom, [vv/t*100 if t!=0 else 0 for vv, t in zip(v, total_time)])
        bottom_compare = np.add(bottom_compare, [vv/t*100 if t!=0 else 0 for vv, t in zip(compare_breakdown[k], total_time_compare)])

    ax.legend(fontsize=4)
    plt.xticks([val for pair in zip([xx-shift for xx in x], [xx+shift for xx in x]) for val in pair], ["w/", "w/o"]*steps_len, fontsize=6)

    sec = ax.secondary_xaxis(location=0)
    if len(xlabel) == 20:
        sec.set_xticks(x, labels=[xl+"        " for xl in xlabel], fontsize=6, rotation=90)
    else:
        sec.set_xticks(x, labels=["\n"+xl for xl in xlabel], fontsize=6)
    sec.tick_params(bottom = False) 

    folder_name_split = args.json_file.split("/")
    title_name = folder_name_split[-1] + " " + exp_name + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n" + ("Warmup with 5 examples, profile result for 6th inference sample" if "t2i" in args.json_file else "Warmup with 10 examples, profile result for 11th inference sample")
    wrapup_graph(plt, ax, exp_name, xlabel, 'Execution Time Breakdown (%)', title_name, save_folder_path, "_decoding_step_operator_breakdown_overall_ratio.pdf", dpi, nested=True)

def graph_overall_grouped(kernel_breakdown, xlabel, exp_name, save_folder_path, secondary_xlabel):
    fig, ax, dpi = prep_graph(nested=True)

    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [[0]*len(secondary_xlabel)]*steps_len
    x = np.arange(steps_len)


    from matplotlib import colormaps
    colormap = colormaps['Set3'].colors

    x = np.arange(len(secondary_xlabel))
    shift = 0.12

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        for idx2, bs in enumerate(xlabel):
            ax.bar([xx-shift*(3-idx2) if idx2<=3 else xx+shift*(idx2-3) for xx in x], v[idx2::len(xlabel)] if len(kernel_breakdown) != 0 else [0]*len(x), bottom=bottom[idx2], label=k if idx2==0 else None, color=cmap[k], width=shift)
            bottom[idx2] = np.add(bottom[idx2], v[idx2::len(xlabel)])


    plt.xticks([val for pair in zip([xx-shift*3 for xx in x], [xx-shift*2 for xx in x], [xx-shift*1 for xx in x], [xx-shift*0 for xx in x], [xx+shift*1 for xx in x], [xx+shift*2 for xx in x], [xx+shift*3 for xx in x]) for val in pair], xlabel*len(secondary_xlabel), fontsize=6)

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(x, labels=["\n\n"+sx for sx in secondary_xlabel], fontsize=6)#, rotation=90)

    # Rotating X-axis labels
    ax.tick_params(axis='x')#, rotation=90)
    ax.set_ylabel("End-to-End Inference Runtime (ms)")

    # ax.legend(loc="right", fontsize=6)
    ax.legend(loc='best', ncol=3, fontsize=6)#, bbox_to_anchor=(0.5, 1.05))

    folder_name_split = args.json_file.split("/")
    title_name = folder_name_split[-1] + " " + re.sub("\n", "", exp_name) + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n" + ("Warmup with 5 examples, profile result for 6th inference sample" if "t2i" in args.json_file else "Warmup with 10 examples, profile result for 11th inference sample")
    wrapup_graph(plt, ax, exp_name, xlabel, 'Execution Time Breakdown (ms)', title_name, save_folder_path, "_decoding_step_operator_breakdown_overall.pdf", dpi)


def gather_result(profile_result, overall_breakdown, add_dummy=0, nested=False):
    if profile_result != dict():
        if len(overall_breakdown)==0:
            for k in profile_result.keys():
                overall_breakdown[k] = [0]*add_dummy if add_dummy > 0 else list()

        assert profile_result.keys() == overall_breakdown.keys()
        for k, v in profile_result.items():
            overall_breakdown[k].append(sum(v))
    else:
        if len(overall_breakdown)==0:
            return add_dummy+1
        else:
            for k, v in overall_breakdown.items():
                overall_breakdown[k].append(0)
    return 0

if args.json_file.split("/")[-1] == "":
    print("Please remove \"/\" at the end of --json-file argument")
    exit(0)

if args.batch_size and not args.multigpu:
    assert "retrieval" not in args.json_file
    # Batch Iterate
    BATCH_SIZE=["1", "4", "8", "16", "32"]
    overall_breakdown = dict()
    for bs in BATCH_SIZE:
        exp_info = args.json_file.split("/")[-1].split("bs")
        file_path = '/'.join(args.json_file.split("/")[:-1])+"/"+exp_info[0]+"bs"+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"/profile.json"
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, overall_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = args.graph_path+"/"+ ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "") + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")+exp_info[0]+'.'.join(exp_info[1].split(".")[1:])+"/"
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall(overall_breakdown, BATCH_SIZE, "batch_size", save_folder_path)
    graph_overall_ratio(overall_breakdown, BATCH_SIZE, "batch_size", save_folder_path)

elif args.n_retrieved_doc:
    assert "retrieval" in args.json_file

    # NRETRIEVED_DOCS Iterate
    NRETRIEVED_DOCS=["1", "2", "3", "4"]
    overall_breakdown = dict()
    for nd in NRETRIEVED_DOCS:
        file_path=args.json_file+".n_retrieved_docs"+nd+"/profile.json"
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, overall_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = args.graph_path+"/"+ ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "") + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")+exp_info[0]+'.'.join(exp_info[1].split(".")[1:])+"/"
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall(overall_breakdown, NRETRIEVED_DOCS, "n_retrieved_docs", save_folder_path)
    graph_overall_ratio(overall_breakdown, NRETRIEVED_DOCS, "n_retrieved_docs", save_folder_path)

elif args.both and not args.compare_efficient_attn:
    assert "retrieval" in args.json_file

    # BATCH_SIZE and NRETRIEVED_DOCS Iterate
    BATCH_SIZE=["bs1", "bs4", "bs8", "bs16"]#, "bs32"]
    NRETRIEVED_DOCS=["n_retrieved_docs1", "n_retrieved_docs2", "n_retrieved_docs3", "n_retrieved_docs4"]
    overall_breakdown = dict()
    for bs in BATCH_SIZE:
        for nd in NRETRIEVED_DOCS:
            exp_info = args.json_file.split("/")[-1].split("bs")
            file_path = '/'.join(args.json_file.split("/")[:-1])+"/"+exp_info[0]+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"."+nd+"/profile.json"
            profile_result = parse_file(file_path, plot_graph=False)
            gather_result(profile_result, overall_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = args.graph_path+"/"+ ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "") + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")+exp_info[0]+'.'.join(exp_info[1].split(".")[1:])+"/"
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall(overall_breakdown, ['.'.join(ip) for ip in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))], "batch_size&n_retrieved_docs", save_folder_path)
    graph_overall_ratio(overall_breakdown, ['.'.join(ip) for ip in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))], "batch_size&n_retrieved_docs", save_folder_path)

elif args.compare_efficient_attn and not args.both:
    assert args.compare_dir != ""
    # Batch Iterate
    BATCH_SIZE=["1", "4", "8", "16", "32"]
    overall_breakdown = dict()
    compare_breakdown = dict()
    for bs in BATCH_SIZE:
        exp_info = args.json_file.split("/")[-1].split("bs")
        file_path = '/'.join(args.json_file.split("/")[:-1])+"/"+exp_info[0]+"bs"+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"/profile.json"
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, overall_breakdown)

        file_path = args.compare_dir+"/"+exp_info[0]+"bs"+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"/profile.json"
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, compare_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = args.graph_path+"/wo_efficient_attn/compare/" + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")+exp_info[0]+'.'.join(exp_info[1].split(".")[1:])+"/"
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall_compare(overall_breakdown, compare_breakdown, BATCH_SIZE, "batch_size", save_folder_path)
    graph_overall_ratio_compare(overall_breakdown, compare_breakdown, BATCH_SIZE, "batch_size", save_folder_path)

elif args.compare_efficient_attn and args.both:
    assert args.compare_dir != ""
    assert "retrieval" in args.json_file
    
    # Batch Iterate
    BATCH_SIZE=["bs1", "bs4", "bs8", "bs16"]#, "bs32"]
    NRETRIEVED_DOCS=["original", "n_retrieved_docs1", "n_retrieved_docs2", "n_retrieved_docs3", "n_retrieved_docs4"]

    overall_breakdown = dict()
    compare_breakdown = dict()
    for bs in BATCH_SIZE:
        for nd in NRETRIEVED_DOCS:
            if nd == "original":
                original_path = re.sub('default_retrieval_template.', 'cm3v2_template.' if "t2i" not in args.json_file else '', re.sub('flamingo_retrieval_v2_template.', 'cm3v2_template.' if "t2i" not in args.json_file else '', args.json_file))
                exp_info = original_path.split("/")[-1].split("bs")
                file_path = '/'.join(original_path.split("/")[:-2])+"/"+exp_info[0]+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"/profile.json"
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, overall_breakdown)

                file_path = re.sub(r'retrieval', '', re.sub(r'flamingo_retrieval_v2_template.', 'cm3v2_template.' if "t2i" not in args.json_file else '', args.compare_dir))+"/"+exp_info[0]+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"/profile.json"
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, compare_breakdown)
            
            else:
                exp_info = args.json_file.split("/")[-1].split("bs")
                file_path = '/'.join(args.json_file.split("/")[:-1])+"/"+exp_info[0]+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"."+nd+"/profile.json"
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, overall_breakdown)

                file_path = args.compare_dir+"/"+exp_info[0]+bs+"."+'.'.join(exp_info[1].split(".")[1:])+"."+nd+"/profile.json"
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, compare_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = args.graph_path+"/wo_efficient_attn/compare/" + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")+exp_info[0]+'.'.join(exp_info[1].split(".")[1:])+"/"
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall_compare(overall_breakdown, compare_breakdown, ['.'.join(l) for l in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))], "batch_size", save_folder_path)
    graph_overall_ratio_compare(overall_breakdown, compare_breakdown, ['.'.join(l) for l in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))], "batch_size", save_folder_path)


elif args.multigpu and args.batch_size:
    NGPU_NNODE=[ ("1-8"), ("1-1"), ("4-1"), ("8-1"), ("4-2"), ("2-4")]
    BATCH_SIZE=["64"]
    # BATCH_SIZE=["1", "4", "8", "16", "32", "64", "128"]
    overall_breakdown = dict()
    overall_latency_breakdown = dict()
    warmup=18
    num_sample=5
    add_dummy=0
    add_dummy2=0
    for config in NGPU_NNODE:
        ngpu, nnode = config.split('-')
        file_dir = re.sub(r"[0-9]gpu_[0-9]node", ngpu+"gpu_"+nnode+"node", args.json_file)
        for bs in BATCH_SIZE:
            multigpu_breakdown = dict()
            multigpu_latency_breakdown = dict()
            all_exist=True

            for sample_id in range(warmup, warmup+num_sample):
                sample_multigpu_breakdown = dict()
                sample_multigpu_latency_breakdown = dict()

                for g in range(int(ngpu)*int(nnode)):
                    exp_info = file_dir.split("/")[-1].split("mbs.")
                    file_path = '/'.join(file_dir.split("/")[:-1])+"/"+exp_info[0]+"mbs."+bs+"."+".".join(exp_info[1].split(".")[1:])+"/profile_sample_"+str(sample_id)+"_gpu_"+str(g)+".json"
                    if not os.path.isfile(file_path):
                        print(file_path)
                    profile_result = parse_file(file_path, plot_graph=False)
                    # profile_result = {'MODULE_PREPROC_ENCODE_IMAGES_AG': [19.204000000000004, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Misc': [8.792000000000002, 0.43700000000000033, 0.4300000000000003, 0.4280000000000003, 0.4240000000000003, 0.4240000000000003, 0.4270000000000003, 0.4310000000000003, 0.4300000000000003, 0.4260000000000003], 'MODULE_ParallelEmbedding_AG': [0.092, 0.003, 0.003, 0.003, 0.004, 0.004, 0.003, 0.004, 0.004, 0.003], 'MODULE_FusedRMSNorm_AG': [4.769000000000001, 3.2909999999999955, 3.2749999999999955, 3.258999999999997, 3.2539999999999956, 3.254999999999996, 3.2509999999999972, 3.251999999999996, 3.2459999999999964, 3.258999999999996], 'MODULE_LayerNorm_AG': [8.097999999999997, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024], 'MODULE__InnerAttention_AG': [25.89500000000002, 6.263999999999969, 6.218999999999967, 6.200999999999968, 6.17399999999997, 6.200999999999968, 6.207999999999968, 6.209999999999968, 6.206999999999969, 6.2039999999999695], 'Copy': [0.35, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004], 'MODULE_SCORING_AG': [0.06300000000000001, 0.08200000000000002, 0.08400000000000002, 0.08200000000000002, 0.08300000000000002, 0.08300000000000002, 0.08400000000000003, 0.08400000000000002, 0.08100000000000002, 0.08500000000000002], 'Communication': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Linear': [314.0490000000001, 40.52999999999994, 40.47799999999992, 40.448999999999934, 40.46899999999994, 40.45599999999993, 40.43899999999994, 40.48199999999996, 40.44099999999993, 40.44599999999994], 'Idle': [13.661999999999864, 78.36300000000011, 77.53000000000011, 80.00700000000012, 76.6300000000001, 80.6880000000001, 76.9760000000001, 80.09600000000007, 77.9250000000001, 80.33800000000008]}
                    if len(profile_result)==0:
                        all_exist = False
                    gather_result(profile_result, sample_multigpu_breakdown)
                    gather_result({k:[vv/int(bs) for vv in v] for k, v in profile_result.items()}, sample_multigpu_latency_breakdown)

                for k, v in sample_multigpu_breakdown.items():
                    assert len(v)==int(ngpu)*int(nnode)
                    assert len(sample_multigpu_latency_breakdown[k])==int(ngpu)*int(nnode)

                if all_exist:
                    sample_multigpu_breakdown = {k: [np.average(v)] for k, v in sample_multigpu_breakdown.items()}
                    sample_multigpu_latency_breakdown = {k: [np.average(v)] for k, v in sample_multigpu_latency_breakdown.items()}
                else:
                    sample_multigpu_breakdown = dict()
                    sample_multigpu_latency_breakdown = dict()

                for k, v in sample_multigpu_breakdown.items():
                    assert len(v)==1, len(v)
                for k, v in sample_multigpu_latency_breakdown.items():
                    assert len(v)==1, len(v)
                    
                gather_result(sample_multigpu_breakdown, multigpu_breakdown)
                gather_result(sample_multigpu_latency_breakdown, multigpu_latency_breakdown)

            for k, v in multigpu_breakdown.items():
                assert len(v)==num_sample, len(v)
            for k, v in multigpu_latency_breakdown.items():
                assert len(v)==num_sample, len(v)

            multigpu_breakdown = {k: [np.average(v)] for k, v in multigpu_breakdown.items()}
            multigpu_latency_breakdown = {k: [np.average(v)] for k, v in multigpu_latency_breakdown.items()}

            add_dummy = gather_result(multigpu_breakdown, overall_breakdown, add_dummy)
            add_dummy2 = gather_result(multigpu_latency_breakdown, overall_latency_breakdown, add_dummy2)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("mbs")
    save_folder_path = args.graph_path+"/"+ ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "") + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")+exp_info[0]+'.'.join(exp_info[1].split(".")[1:])+"/"
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall_grouped(overall_breakdown, BATCH_SIZE, "\n\nbatch_size", save_folder_path, secondary_xlabel=NGPU_NNODE)
    save_folder_path = args.graph_path+"/"+ ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "") + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")+exp_info[0]+'.'.join(exp_info[1].split(".")[1:])+"/latency/"
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall_grouped(overall_latency_breakdown, BATCH_SIZE, "\n\nbatch_size", save_folder_path, secondary_xlabel=NGPU_NNODE)

else:
    folder_name_split = args.json_file.split("/")
    # exp_info = folder_name_split[-1].split("bs")
    # save_folder_path = args.graph_path+"/"+(folder_name_split[-2:] if "retrieval" in args.json_file else folder_name_split[-1])
    save_folder_path = args.graph_path+"/"+ ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "") + ("retrieval/" if "retrieval" in args.json_file else "")+folder_name_split[-2]+"/"
    profile_result = parse_file(args.json_file, save_folder_path=save_folder_path)
