"""
Usage: python torch_profiler_parser.py <path to directory of traces> <prefix>
Example: python torch_profiler_parser.py /private/home/yejinlee/profiler_results/profiler_trace_tb_torch2.1_SPEECH_TO_SPEECH_module_breakdown_8gpu_batch_size1 YJ_PROFILE
"""

import argparse
import csv
import glob
import itertools

# %%
import json

import os, sys
import pickle
import re
from multiprocessing import freeze_support, Manager, Pool, RLock

import numpy as np

from matplotlib import colormaps, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from tqdm import tqdm

DIR_PREFIX = "./onellm_scripts/data_for_paper/pickle_dumps/"

# Initialize ArgParser:
parser = argparse.ArgumentParser("argparser")

# Parse possible arguments:
parser.add_argument("--json-file", type=str, default="profile.json")
parser.add_argument("--json-folder", type=str, default="")
parser.add_argument(
    "--desired-prefixes", type=str, default="RESBLOCK_AG*ATTENTION_AG*FEEDFORWARD_AG"
)
parser.add_argument(
    "--graph-path", type=str, default="/fsx-checkpoints/yejinlee/analysis_figures"
)
parser.add_argument("--batch-size", action="store_true", default=False)
parser.add_argument("--n-retrieved-doc", action="store_true", default=False)
parser.add_argument("--both", action="store_true", default=False)
parser.add_argument("--compare-efficient-attn", action="store_true", default=False)
parser.add_argument("--compare-dir", type=str, default="")
parser.add_argument("--multigpu", action="store_true", default=False)
parser.add_argument("--simplify", action="store_true", default=False)
parser.add_argument("--figure1", action="store_true", default=False)
parser.add_argument("--figure1_separate", action="store_true", default=False)
parser.add_argument("--export", action="store_true", default=False)
parser.add_argument("--import_pickle", action="store_true", default=False)


# Get arguments into readable format:
args = parser.parse_args()
print("Finish parsing arguments")
print("args = ", args)

FONTSIZE = 6 if args.figure1 or args.figure1_separate else 26

prefix = "AG"

# desired_prefixes = ["RESBLOCK_AG", "ATTENTION_AG"]
# desired_prefixes = ['MODULE_Embedding_AG', 'MODULE_LayerNorm_AG', 'MODULE_Linear_AG', 'MODULE_QuickGELUActivation_AG', 'MODULE_Timesteps_AG', 'MODULE_SiLU_AG', 'MODULE_Conv2d_AG', 'MODULE_GroupNorm_AG', 'MODULE_Dropout_AG', 'MODULE_Attention_AG']

desired_prefixes = list(set(args.desired_prefixes.split("*")))

top_prefixes = ["DENOISING_LOOP"]


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


def process_kernel_breakdown(
    kernel_breakdown,
    decoding_step_time,
    gpu_operation_time_per_decoding_step,
    max_kernel,
    min_kernel,
    plot_graph,
):

    if (
        "MODULE_ColumnParallelLinear_AG" in kernel_breakdown
        and "MODULE_RowParallelLinear_AG" in kernel_breakdown
    ):
        kernel_breakdown["Linear"] = [
            a + b
            for a, b in zip(
                kernel_breakdown["MODULE_ColumnParallelLinear_AG"],
                kernel_breakdown["MODULE_RowParallelLinear_AG"],
            )
        ]
        del kernel_breakdown["MODULE_ColumnParallelLinear_AG"]
        del kernel_breakdown["MODULE_RowParallelLinear_AG"]

    if "MODULE_ToMe_Merge_AG" in kernel_breakdown:
        kernel_breakdown["MODULE_ToMe_Merge_AG"] = [0] * (
            len(kernel_breakdown["Misc"])
            - len(kernel_breakdown["MODULE_ToMe_Merge_AG"])
        ) + kernel_breakdown["MODULE_ToMe_Merge_AG"]

    if "MODULE_LOG_SOFTMAX_AG" in kernel_breakdown:
        kernel_breakdown["MODULE_SCORING_AG"] = kernel_breakdown.pop(
            "MODULE_LOG_SOFTMAX_AG"
        )

    # if sum(communication_list) > 0:
    #     kernel_breakdown["Communication"] = communication_list

    new_kernel_breakdown = dict()
    for k, v in kernel_breakdown.items():
        new_kernel_breakdown[
            re.sub(
                "POST_PROC_IMAGE_DECODE",
                "(Postprocessing) Decode Image",
                re.sub(
                    "POSTPROC_GENERATE_TEXT",
                    "(Postprocessing) Generate Text",
                    (
                        re.sub(
                            "PREPROC_ENCODE_IMAGES",
                            "(Preprocessing) Encode Image",
                            re.sub(
                                "SCORING",
                                "Scoring",
                                re.sub("_AG", "", re.sub("MODULE_", "", k)),
                            ),
                        )
                    ),
                ),
            )
        ] = v
    del kernel_breakdown
    kernel_breakdown = new_kernel_breakdown

    def merge_values(kernel_breakdown, org_k, new_k):
        assert org_k != new_k
        if org_k in kernel_breakdown:
            if new_k in kernel_breakdown:
                kernel_breakdown[new_k] = [
                    a + b
                    for a, b in zip(kernel_breakdown[new_k], kernel_breakdown[org_k])
                ]
                del kernel_breakdown[org_k]
            else:
                kernel_breakdown[new_k] = kernel_breakdown.pop(org_k)
        return kernel_breakdown

    kernel_breakdown = merge_values(
        kernel_breakdown, "LlamaRotaryEmbedding", "Embedding"
    )
    kernel_breakdown = merge_values(kernel_breakdown, "ParallelEmbedding", "Embedding")
    kernel_breakdown = merge_values(kernel_breakdown, "StandardEmbedding", "Embedding")
    kernel_breakdown = merge_values(kernel_breakdown, "_InnerAttention", "Attention")
    kernel_breakdown = merge_values(kernel_breakdown, "TorchSDPA", "Attention")
    # kernel_breakdown = merge_values(kernel_breakdown, "SequentialTransductionUnit", "Attention")
    kernel_breakdown = merge_values(
        kernel_breakdown, "_RaggedAttentionRelativeBiasFunction", "Attention"
    )
    kernel_breakdown = merge_values(kernel_breakdown, "Sigmoid", "Activation")
    kernel_breakdown = merge_values(kernel_breakdown, "GLU", "Activation")
    kernel_breakdown = merge_values(kernel_breakdown, "ReLU", "Activation")
    kernel_breakdown = merge_values(kernel_breakdown, "SiLU", "Activation")
    kernel_breakdown = merge_values(kernel_breakdown, "StandardLayerNorm", "LayerNorm")
    kernel_breakdown = merge_values(kernel_breakdown, "FusedRMSNorm", "LayerNorm")
    kernel_breakdown = merge_values(kernel_breakdown, "LlamaRMSNorm", "LayerNorm")
    kernel_breakdown = merge_values(kernel_breakdown, "RMSNorm", "LayerNorm")
    kernel_breakdown = merge_values(kernel_breakdown, "TiedProjection", "Linear")
    kernel_breakdown = merge_values(kernel_breakdown, "ConvTranspose1d", "Conv1d")

    if plot_graph:
        graph_gpu_kernel_breakdown(kernel_breakdown, save_folder_path)

    idle_list = list()
    if len(decoding_step_time) > 0:
        print(len(kernel_breakdown["Scoring"]))
        for s in np.arange(len(kernel_breakdown["Scoring"])):
            print(
                "Step ",
                s,
                decoding_step_time[s],
                " / ",
                gpu_operation_time_per_decoding_step[s],
            )
            assert gpu_operation_time_per_decoding_step[s] >= 0
            idle_list.append(gpu_operation_time_per_decoding_step[s])
    else:
        # end_kernel = gpu_kernels_dict[gpu_launch_kernels[-1]['args']['External id']]
        end_kernel = max_kernel
        # print("END", end_kernel)
        # print("START", gpu_launch_kernels[0])
        # print((end_kernel['ts']+end_kernel['dur']-gpu_launch_kernels[0]['ts'])/1000)

        # assert (end_kernel['ts']+end_kernel['dur']-gpu_launch_kernels[0]['ts'])/1000-sum([sum(v) for v in kernel_breakdown.values()]) >= 0
        # idle_list.append((end_kernel['ts']+end_kernel['dur']-gpu_launch_kernels[0]['ts'])/1000-sum([sum(v) for v in kernel_breakdown.values()]))
        assert (end_kernel["ts"] + end_kernel["dur"] - min_kernel["ts"]) / 1000 - sum(
            [sum(v) for v in kernel_breakdown.values()]
        ) >= 0
        idle_list.append(
            (end_kernel["ts"] + end_kernel["dur"] - min_kernel["ts"]) / 1000
            - sum([sum(v) for v in kernel_breakdown.values()])
        )

    kernel_breakdown["Idle"] = idle_list

    if plot_graph:
        graph_gpu_kernel_breakdown_idle(kernel_breakdown, save_folder_path)

    kernel_breakdown = merge_values(kernel_breakdown, "Scoring", "Misc")
    kernel_breakdown = merge_values(kernel_breakdown, "Copy", "Misc")
    kernel_breakdown = merge_values(
        kernel_breakdown, "(Preprocessing) Encode Image", "Misc"
    )
    kernel_breakdown = merge_values(
        kernel_breakdown, "(Postprocessing) Decode Image", "Misc"
    )
    kernel_breakdown = merge_values(kernel_breakdown, "Activation", "Misc")
    kernel_breakdown = merge_values(
        kernel_breakdown, "Wav2Vec2FbankFeatureExtractor", "Misc"
    )
    kernel_breakdown = merge_values(kernel_breakdown, "Masked_Select", "Misc")
    kernel_breakdown = merge_values(kernel_breakdown, "HardUpsampling", "Misc")
    kernel_breakdown = merge_values(
        kernel_breakdown, "SinusoidalPositionEncoder", "Misc"
    )

    return kernel_breakdown


def parse_file(file_path, save_folder_path=None, plot_graph=True):
    kernel_breakdown = {}
    desired_prefixes_gpu_dur = []

    if os.path.isfile(file_path):
        file_p = open(file_path)
        print("Reading from ", file_path)
    else:
        print("File doesn't exists!!!! ", file_path)
        return kernel_breakdown

    print("NOW DESIRED PREFIXES = ", desired_prefixes)

    print("PARSING NOW ")
    jsonString = json.load(file_p)["traceEvents"]

    slices = dict()
    total_train_time = 0

    cpu_desired_ops = []
    cpu_top_ops = []

    gpu_launch_kernels = []

    gpu_kernels = []

    nccl_kernels = []

    with tqdm(total=len(jsonString), desc="Profiling...", position=0) as pbar:
        ####################### FIRST LOOP THROUGH JSON #################################
        for idx, l in enumerate(jsonString):
            found_desired_prefix = False

            # IF ITS A CPU KERNEL:
            if (
                "cat" in l
                and (l["cat"] == "cpu_op" or l["cat"] == "user_annotation")
                and "ts" in l
                and "dur" in l
            ):

                for p in desired_prefixes:
                    if p in l["name"]:
                        cpu_desired_ops.append(l)
                        found_desired_prefix = True

                if found_desired_prefix == False:
                    for top in top_prefixes:
                        if top in l["name"]:
                            cpu_top_ops.append(l)

            # IF ITS A GPU KERNEL:
            if "cat" in l and l["cat"] == "kernel":

                # If there is not a flow event right after this, error
                if jsonString[idx + 1]["ph"] != "f":
                    print("ERROR")
                    exit(0)

                if "nccl" in l["name"]:
                    nccl_kernels.append(l)
                else:
                    found = False
                    # gpu_kernels.append(l)
                    # gpu_launch_kernels.append(jsonString[idx+2])

                    # assert jsonString[idx+2]["name"] == "cudaLaunchKernel"
                    # assert l["args"]["External id"] == jsonString[idx+2]["args"]["External id"]

                    for ii in range(
                        idx + 2, len(jsonString)
                    ):  # should find the launch kernel closest to this event
                        if (
                            jsonString[ii]["name"] == "cudaLaunchKernel"
                        ):  # if there is a launch kernel there
                            gpu_launch_kernels.append(jsonString[ii])  # From(cpu)
                            gpu_kernels.append(
                                l
                            )  # To(gpu kernel) # append this to the GPU kernel
                            found = True
                            break
                    if found == False:
                        print("ERROR2")
                        for ii in range(
                            0, len(jsonString)
                        ):  # should find the launch kernel closest to this event
                            if (
                                jsonString[ii]["name"] == "cudaLaunchKernel"
                            ):  # if there is a launch kernel there
                                gpu_launch_kernels.append(jsonString[ii])  # From(cpu)
                                gpu_kernels.append(
                                    l
                                )  # To(gpu kernel) # append this to the GPU kernel
                                found = True
                                break

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
    current_cpu_desired_index_changed = False

    print("Start sorting")

    gpu_launch_kernels = sorted(gpu_launch_kernels, key=lambda x: x["ts"])
    # gpu_kernels = sorted(gpu_kernels, key=lambda x: x['ts'])
    gpu_kernels_dict = dict()
    for gk in gpu_kernels:
        gpu_kernels_dict[gk["args"]["External id"]] = gk

    nccl_kernels = sorted(nccl_kernels, key=lambda x: x["ts"])
    cpu_desired_ops = sorted(cpu_desired_ops, key=lambda x: x["ts"])

    print("Finished sorting")
    # print("Min gpu kernels: ", gpu_kernels[0])
    # print("Max gpu kernels: ", gpu_kernels[-1])
    # if len(nccl_kernels) > 0:
    #     print("Min nccl kernels: ", nccl_kernels[0])
    #     print("Max nccl kernels: ", nccl_kernels[-1])

    # min_launch = gpu_kernels[0]
    min_launch = None
    # max_launch = gpu_kernels[0]
    max_launch = gpu_kernels_dict[gpu_launch_kernels[0]["args"]["External id"]]
    decoding_step_time = list()
    gpu_operation_time = 0
    gpu_operation_time_per_decoding_step = list()
    end_of_decoding_step = None
    entered_scoring = False
    decoding_step_start_time = [
        gpu_kernels_dict[gpu_launch_kernels[0]["args"]["External id"]]["ts"]
    ]

    test = set()
    test2 = set()
    # for idx,l in enumerate(jsonString):
    for idx, launch in tqdm(enumerate(gpu_launch_kernels)):

        # print("==========================================================================")

        time_start = launch["ts"]
        time_end = launch["ts"] + launch["dur"]

        # print("time_start = ", time_start)
        # print("time_end = ", time_end)

        # corresp_gpu_kernel = gpu_kernels[idx]
        # print(launch)
        corresp_gpu_kernel = gpu_kernels_dict[launch["args"]["External id"]]
        assert (
            launch["args"]["External id"] == corresp_gpu_kernel["args"]["External id"]
        )

        gpu_time_start = corresp_gpu_kernel["ts"]
        gpu_time_end = corresp_gpu_kernel["ts"] + corresp_gpu_kernel["dur"]

        if gpu_time_end >= max_launch["ts"]:
            max_launch = corresp_gpu_kernel

        found_cpu = False

        while current_cpu_desired_index < len(cpu_desired_ops) and found_cpu == False:
            cpu_op = cpu_desired_ops[current_cpu_desired_index]

            cpu_start = cpu_op["ts"]
            cpu_end = cpu_op["ts"] + cpu_op["dur"]

            if entered_scoring == False:
                # end_of_decoding_step = gpu_kernels[idx-1]
                end_of_decoding_step = gpu_kernels_dict[
                    gpu_launch_kernels[idx - 1]["args"]["External id"]
                ]

            # In annotation
            if cpu_start <= time_start and cpu_end >= time_end:
                name = cpu_op["name"]

                if (
                    "ParallelEmbedding" in name or "Embedding" in name
                ) and min_launch == None:
                    min_launch = corresp_gpu_kernel

                if name in kernel_breakdown.keys():
                    kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"] / 1000
                else:
                    kernel_breakdown[name] = [corresp_gpu_kernel["dur"] / 1000]

                found_cpu = True
                if name == "MODULE_SCORING_AG" or name == "MODULE_LOG_SOFTMAX_AG":
                    entered_scoring = True
                else:
                    if min_launch != None:
                        # print(name)
                        gpu_operation_time += corresp_gpu_kernel["dur"] / 1000
                    # else:
                    #     print(name, " ", corresp_gpu_kernel)
            # past end of annotation
            elif cpu_end <= time_start:
                # if current_cpu_desired_index_changed==True and cpu_op["name"] == "MODULE_SCORING_AG":
                if current_cpu_desired_index < len(cpu_desired_ops) and (
                    cpu_op["name"] == "MODULE_SCORING_AG"
                    or cpu_op["name"] == "MODULE_LOG_SOFTMAX_AG"
                ):
                    # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"])
                    # print(min_launch, " ", gpu_kernels[idx-1])

                    # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"], " / ", end_of_decoding_step["ts"]+end_of_decoding_step["dur"]-min_launch["ts"])
                    # print(min_launch, " ", end_of_decoding_step, " / idx-1: ", idx-1)
                    gpu_operation_time_per_decoding_step.append(gpu_operation_time)
                    gpu_operation_time = 0
                    decoding_step_time.append(
                        (
                            end_of_decoding_step["ts"]
                            + end_of_decoding_step["dur"]
                            - min_launch["ts"]
                        )
                        / 1000
                    )
                    end_of_decoding_step = None
                    # min_launch = max_launch
                    min_launch = None
                    entered_scoring = False

                    for k, v in kernel_breakdown.items():
                        v.append(0)
                    decoding_step_start_time.append(gpu_time_start)

                # if also before end of next annotation
                if (
                    current_cpu_desired_index == len(cpu_desired_ops) - 1
                    or time_end <= cpu_desired_ops[current_cpu_desired_index + 1]["ts"]
                ):
                    name = "Copy" if "copy" in corresp_gpu_kernel["name"] else "Misc"
                    if name in kernel_breakdown.keys():
                        kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"] / 1000
                    else:
                        kernel_breakdown[name] = [corresp_gpu_kernel["dur"] / 1000]

                    if min_launch != None:
                        # print(name)
                        gpu_operation_time += corresp_gpu_kernel["dur"] / 1000
                    # else:
                    #     print(name, " ", corresp_gpu_kernel)

                    found_cpu = True
                current_cpu_desired_index = current_cpu_desired_index + 1

            elif time_end <= cpu_start:
                name = "Copy" if "copy" in corresp_gpu_kernel["name"] else "Misc"

                if name in kernel_breakdown.keys():
                    kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"] / 1000
                else:
                    kernel_breakdown[name] = [corresp_gpu_kernel["dur"] / 1000]

                if min_launch != None:
                    # print(name)
                    gpu_operation_time += corresp_gpu_kernel["dur"] / 1000
                # else:
                #     print(name, " ", corresp_gpu_kernel)

                found_cpu = True
            else:
                # This should not happen, but happend in
                print("[ALERTALERTALERTALERTALERTALERTALERTALERTALERT]")
                name = cpu_op["name"]

                if name in kernel_breakdown.keys():
                    kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"] / 1000
                else:
                    kernel_breakdown[name] = [corresp_gpu_kernel["dur"] / 1000]

                if min_launch != None:
                    # print(name)
                    gpu_operation_time += corresp_gpu_kernel["dur"] / 1000
                # else:
                #     print(name, " ", corresp_gpu_kernel)

                found_cpu = True
                if name == "MODULE_SCORING_AG" or name == "MODULE_LOG_SOFTMAX_AG":
                    # print("SCORING ", corresp_gpu_kernel)
                    entered_scoring = True
                else:
                    if min_launch != None:
                        # print(name)
                        gpu_operation_time += corresp_gpu_kernel["dur"] / 1000
                    # else:
                    #     print(name, " ", corresp_gpu_kernel)

        # For ones at the end that haven't been assigned but are past the cpu annotations
        if found_cpu == False:
            name = "Copy" if "copy" in corresp_gpu_kernel["name"] else "Misc"
            if name in kernel_breakdown.keys():
                kernel_breakdown[name][-1] += corresp_gpu_kernel["dur"] / 1000
            else:
                kernel_breakdown[name] = [corresp_gpu_kernel["dur"] / 1000]

            if min_launch != None:
                # print(name)
                gpu_operation_time += corresp_gpu_kernel["dur"] / 1000
            # else:
            #     print(name, " ", corresp_gpu_kernel)

    if (
        cpu_op["name"] == "MODULE_SCORING_AG"
        or cpu_op["name"] == "MODULE_LOG_SOFTMAX_AG"
    ):
        if end_of_decoding_step is not None and min_launch is not None:
            # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"])
            # print(min_launch, " ", corresp_gpu_kernel)

            # print("Min: ", min_launch["ts"], " Max: ", end_of_decoding_step["ts"], " / ", end_of_decoding_step["ts"]+end_of_decoding_step["dur"]-min_launch["ts"])
            # print(min_launch, " ", end_of_decoding_step, " / idx-1: ", idx-1)

            gpu_operation_time_per_decoding_step.append(gpu_operation_time)
            decoding_step_time.append(
                (
                    end_of_decoding_step["ts"]
                    + end_of_decoding_step["dur"]
                    - min_launch["ts"]
                )
                / 1000
            )
        elif end_of_decoding_step is None and min_launch is None:
            gpu_operation_time_per_decoding_step.append(0)
            decoding_step_time.append(0)
        else:
            assert False

    if len(nccl_kernels) > 0:
        communication_list = [0]
        idx = 1
        for nc in nccl_kernels:
            if (
                idx == len(decoding_step_start_time)
                or decoding_step_start_time[idx] >= nc["ts"]
            ):
                communication_list[idx - 1] += nc["dur"] / 1000
            else:
                idx += 1
                if idx < len(decoding_step_start_time):
                    assert decoding_step_start_time[idx] >= nc["ts"]

                communication_list.append(nc["dur"] / 1000)
    else:
        communication_list = [0] * len(decoding_step_start_time)

    print("DECODING STEP TIME: ", decoding_step_time)
    for key, value in kernel_breakdown.items():
        print(key, " = ", value)
    print("end")

    if args.export:
        dump_dir = "./onellm_scripts/data_for_paper/pickle_dumps/" + "/".join(
            file_path.split("/")[:-1]
        )
        os.makedirs(dump_dir, exist_ok=True)
        file_path = dump_dir + "/" + file_path.split("/")[-1] + ".pickle"
        dump_dict = {
            "kernel_breakdown": kernel_breakdown,
            "decoding_step_time": decoding_step_time,
            "gpu_operation_time_per_decoding_step": gpu_operation_time_per_decoding_step,
            "max_kernel": gpu_kernels_dict[
                gpu_launch_kernels[-1]["args"]["External id"]
            ],
            "min_kernel": gpu_launch_kernels[0],
        }
        with open(file=file_path, mode="wb") as f:
            pickle.dump(dump_dict, f)
        print("Written to ", file_path)
        return

    return process_kernel_breakdown(
        kernel_breakdown,
        decoding_step_time,
        gpu_operation_time_per_decoding_step,
        gpu_kernels_dict[gpu_launch_kernels[-1]["args"]["External id"]],
        gpu_launch_kernels[0],
        plot_graph=plot_graph,
    )


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

colormap = colormaps["Set3"].colors
colormap2 = colormaps["Paired"].colors

# cmap = {
#     "Embedding": colormap[0],
#     "Misc": colormap[1],
#     "Linear": colormap[2],
#     "LayerNorm": colormap[3],
#     "Attention": colormap[4],
#     "Copy": colormap[5],
#     "Scoring": colormap[6],
#     "Idle": colormap[7],
#     "Communication": colormap[8],
#     "(Preprocessing) Encode Image": colormap[9],
#     "(Postprocessing) Generate Text": colormap2[0],
#     "(Postprocessing) Decode Image": colormap2[0],
#     "Compute": colormap2[1],
#     "Activation": colormap2[2],
#     "Conv1d": colormap2[3],
#     "SinusoidalPositionEncoder": colormap2[5],
#     "HardUpsampling": colormap2[7],
#     "KV_Cache_Reorder": colormap2[4],
#     "Wav2Vec2FbankFeatureExtractor": colormap2[9],
#     "Masked_Select": colormap2[11],
# }

# Used for paper submission
# cmap = {
#     "Embedding": colormap[0],
#     "Misc": colormap[1],
#     "Linear": colormap[2],
#     "LayerNorm": colormap[3],
#     "Attention": colormap[4],
#     "Copy": colormap[5],
#     "Conv1d": colormap2[6],
#     "Idle": colormap[7],
#     "KV_Cache_Reorder": colormap[10],
# }

# Used for rebuttal
colormap = colormaps["tab10"].colors
colormap2 = colormaps["Set2"].colors
colormap3 = colormaps["Set3"].colors
# colormap3 = colormaps["Set2"].colors

cmap = {
    "Embedding": colormap[0],
    "Misc": colormap3[11],
    "Linear": colormap3[3],
    "LayerNorm": colormap[3],
    "Attention": colormap2[2],
    "Copy": colormap[5],
    "Conv1d": colormap[6],
    "Idle": colormap3[8],
    "KV_Cache_Reorder": colormap3[0],
}

# colormap = colormaps["Accent"].colors

# cmap = {
#     "Embedding": colormap[0],
#     "Misc": colormap[1],
#     "Linear": colormap[2],
#     "LayerNorm": colormap[4],
#     "Attention": colormap[5],
#     "Copy": colormap[6],
#     "Conv1d": colormap[7],
#     "Idle": colormap[8],
#     "KV_Cache_Reorder": colormap[9],
# }


def prep_graph(nested=False):
    print("GRAPHING")

    figures_dir = "."

    dpi = 320

    page_width = 64 if args.multigpu and args.batch_size else 12
    lr_margin = 0.35 if args.multigpu and args.batch_size else 0.75
    column_separation = 0.1 if args.multigpu and args.batch_size else 0.25

    fig_width = (page_width - 2 * lr_margin - column_separation) / 2
    dbl_fig_width = page_width - 2 * lr_margin
    fig_height = (
        0.35 * fig_width
        if args.multigpu and args.batch_size
        else (0.45 * fig_width if not nested else 0.8 * fig_width)
    )

    fig, ax = plt.subplots(
        1, figsize=(fig_width, fig_height), dpi=dpi, layout="tight"
    )  # List of labels
    return fig, ax, dpi


def wrapup_graph(
    plt,
    ax,
    exp_name,
    xlabel,
    yaxis_title,
    title,
    save_folder_path,
    filename,
    dpi,
    nested=False,
    file_name_passed=False,
):
    plt.ylabel(yaxis_title, fontsize=FONTSIZE)
    plt.xlabel(
        ("\n\n\n\n\n\n\n\n\n\n\n" if nested else "") + exp_name, fontsize=FONTSIZE
    )
    # plt.title(title, fontsize=FONTSIZE)

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    ax.tick_params(
        axis="x",
        rotation=(
            90
            if len(xlabel[-1]) > 8 and not args.figure1 and not args.figure1_separate
            else 0
        ),
    )

    plt.grid(lw=0.2)
    ax.set_axisbelow(True)

    file_path = (
        save_folder_path + re.sub(" ", "", re.sub("\n", "", exp_name)) + filename
        if not file_name_passed
        else save_folder_path
    )
    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
    print("graph_path = ", file_path)
    plt.show()


def graph_gpu_kernel_breakdown(kernel_breakdown, save_folder_path):
    fig, ax, dpi = prep_graph()

    steps_len = (
        len(kernel_breakdown["Scoring"])
        if "Scoring" in kernel_breakdown
        else len(kernel_breakdown[list(kernel_breakdown.keys())[0]])
    )
    steps = np.arange(steps_len)
    bottom = [0] * steps_len
    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        print(k)
        ax.bar(steps, v, bottom=bottom, label=k, color=cmap[k], width=0.8)
        bottom = np.add(bottom, v)

    ax.legend(fontsize=4)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-2]
        + "\nOperator Breakdown\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )
    plt.title(title_name, fontsize=6)
    plt.ylabel("Execution Time (ms)", fontsize=6)
    plt.xlabel("Decoding Step", fontsize=6)

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)

    if "retrieval" not in args.json_file:
        if "i2t" in folder_name_split[-2]:
            plt.ylim(0, 2500)
        elif "it2t" in folder_name_split[-2]:
            plt.ylim(0, 2500)
        elif "t2i" in folder_name_split[-2]:
            plt.ylim(0, 1400)
    else:
        if "bs4" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0, 1400)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0, 1400)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0, 1000)
        elif "bs16" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0, 5500)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0, 5500)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0, 1000)

    plt.grid(lw=0.2)
    ax.set_axisbelow(True)
    plt.show()

    # folder_path = args.graph_path+"/"+folder_name_split[-2]
    os.makedirs(save_folder_path, exist_ok=True)
    file_path = save_folder_path + "decoding_step_operator_breakdown.pdf"
    print("graph_path = ", file_path)
    plt.savefig(file_path, dpi=dpi)


def graph_gpu_kernel_breakdown_idle(kernel_breakdown, save_folder_path):
    fig, ax, dpi = prep_graph()

    steps_len = (
        len(kernel_breakdown["Scoring"])
        if "Scoring" in kernel_breakdown
        else len(kernel_breakdown[list(kernel_breakdown.keys())[0]])
    )
    steps = np.arange(steps_len)
    bottom = [0] * steps_len

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(steps, v, bottom=bottom, label=k, color=cmap[k], width=0.8)
        bottom = np.add(bottom, v)

    ax.legend(fontsize=4)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-2]
        + "\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )
    plt.title(title_name, fontsize=6)
    plt.ylabel("Execution Time (ms)", fontsize=6)
    plt.xlabel("Decoding Step", fontsize=6)

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)

    if "retrieval" not in args.json_file:
        if "i2t" in folder_name_split[-2]:
            plt.ylim(0, 2500)
        elif "it2t" in folder_name_split[-2]:
            plt.ylim(0, 2500)
        elif "t2i" in folder_name_split[-2]:
            plt.ylim(0, 1400)
    else:
        if "bs4" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0, 1400)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0, 1400)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0, 1000)
        elif "bs16" in args.json_file:
            if "i2t" in folder_name_split[-2]:
                plt.ylim(0, 5500)
            elif "it2t" in folder_name_split[-2]:
                plt.ylim(0, 5500)
            elif "t2i" in folder_name_split[-2]:
                plt.ylim(0, 1000)

    plt.grid(lw=0.2)
    ax.set_axisbelow(True)
    plt.show()

    # folder_path = args.graph_path+"/"+folder_name_split[-2]
    os.makedirs(save_folder_path, exist_ok=True)
    file_path = save_folder_path + "decoding_step_operator_breakdown_idle.pdf"
    print("graph_path = ", file_path)
    plt.savefig(file_path, dpi=dpi)


def graph_overall(
    kernel_breakdown,
    xlabel,
    exp_name,
    save_folder_path,
    nested=False,
    secondary_xlabel=None,
    file_name_passed=False,
):
    fig, ax, dpi = prep_graph()

    steps_len = len(xlabel)
    bottom = [0] * steps_len

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(xlabel, v, bottom=bottom, label=k, color=cmap[k], width=0.8)
        bottom = np.add(bottom, v)

    ax.legend(fontsize=4, ncol=5, bbox_to_anchor=(0.5, 1.28), loc="upper center")
    # if nested:
    #     sec = ax.secondary_xaxis(location=0)
    #     sec.set_xticks(x, labels=secondary_xlabel, fontsize=6)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-1]
        + " "
        + exp_name
        + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )

    wrapup_graph(
        plt,
        ax,
        exp_name,
        xlabel,
        "Execution Time Breakdown (ms)",
        title_name,
        save_folder_path,
        "",
        dpi,
        nested=nested,
        file_name_passed=file_name_passed,
    )


def graph_overall_ratio(
    kernel_breakdown,
    xlabel,
    exp_name,
    save_folder_path,
    nested=False,
    secondary_xlabel=None,
    file_name_passed=False,
):
    fig, ax, dpi = prep_graph()

    steps_len = len(xlabel)
    bottom = [0] * steps_len

    total_time = list()
    for i in range(steps_len):
        total_time.append(sum([v[i] for v in kernel_breakdown.values()]))

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(
            xlabel,
            [vv / t * 100 if t != 0 else 0 for vv, t in zip(v, total_time)],
            bottom=bottom,
            label=k,
            color=cmap[k],
            width=0.8,
        )
        bottom = np.add(
            bottom, [vv / t * 100 if t != 0 else 0 for vv, t in zip(v, total_time)]
        )

    ax.legend(fontsize=4, ncol=5, bbox_to_anchor=(0.5, 1.43), loc="upper center")

    # if nested:
    #     sec = ax.secondary_xaxis(location=0)
    #     sec.set_xticks(x, labels=secondary_xlabel, fontsize=6)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-1]
        + " "
        + exp_name
        + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )

    wrapup_graph(
        plt,
        ax,
        exp_name,
        xlabel,
        "Execution Time Breakdown (%)",
        title_name,
        save_folder_path,
        "",
        dpi,
        nested=nested,
        file_name_passed=file_name_passed,
    )


def graph_overall_compare(
    kernel_breakdown,
    compare_breakdown,
    xlabel,
    exp_name,
    save_folder_path,
    file_name_passed=False,
):
    fig, ax, dpi = prep_graph(nested=True)

    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [0] * steps_len
    bottom_compare = [0] * steps_len
    x = np.arange(steps_len)

    shift = 0.2

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        # ax.bar(xlabel, v, bottom = bottom, label=k, color=cmap[k], width=0.8)
        ax.bar(
            [xx - shift for xx in x],
            v if len(kernel_breakdown) != 0 else [0] * len(x),
            bottom=bottom,
            label=k,
            color=cmap[k],
            width=0.35,
        )
        ax.bar(
            [xx + shift for xx in x],
            compare_breakdown[k] if len(compare_breakdown) != 0 else [0] * len(x),
            bottom=bottom_compare,
            color=cmap[k],
            width=0.35,
        )
        bottom = np.add(bottom, v)
        bottom_compare = np.add(bottom_compare, compare_breakdown[k])

    ax.legend(fontsize=4)
    plt.xticks(
        [
            val
            for pair in zip([xx - shift for xx in x], [xx + shift for xx in x])
            for val in pair
        ],
        ["w/", "w/o"] * steps_len,
        fontsize=6,
    )

    sec = ax.secondary_xaxis(location=0)
    if len(xlabel) == 20:
        sec.set_xticks(
            x, labels=[xl + "        " for xl in xlabel], fontsize=6, rotation=90
        )
    else:
        sec.set_xticks(x, labels=["\n" + xl for xl in xlabel], fontsize=6)
    sec.tick_params(bottom=False)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-1]
        + " "
        + exp_name
        + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )
    wrapup_graph(
        plt,
        ax,
        exp_name,
        xlabel,
        "Execution Time Breakdown (ms)",
        title_name,
        save_folder_path,
        "_decoding_step_operator_breakdown_overall.pdf",
        dpi,
        nested=True,
        file_name_passed=file_name_passed,
    )


def graph_overall_compare_separate(
    kernel_breakdown,
    compare_breakdown,
    key_order,
    xlabel,
    secondary_xlabel,
    exp_name,
    save_folder_path,
    nested=False,
    file_name_passed=False,
):
    fig, ax, dpi = prep_graph()

    steps_len = len(xlabel)

    # steps = np.arange(steps_len)
    bottom = [0] * steps_len
    bottom_compare = [0] * steps_len
    x = np.arange(steps_len)

    shift = 0.2

    num_separated_bar = 4

    # for idx, (k, v) in enumerate(kernel_breakdown.items()):
    for idx, k in enumerate(key_order):
        label1 = [float(xx) for xx in x.copy()]
        label2 = [float(xx) for xx in x.copy()]

        value1 = list()
        value2 = list()
        for idxidx, vv in enumerate(kernel_breakdown[k]):
            if idxidx < num_separated_bar:
                value1.append(vv)
                value2.append(compare_breakdown[k][idxidx])
                label1[idxidx] -= shift
                label2[idxidx] += shift
            else:
                value1.append(0)
                value2.append(vv)

        ax.bar(label1, value1, bottom=bottom, label=k, color=cmap[k], width=0.35)
        ax.bar(label2, value2, bottom=bottom_compare, color=cmap[k], width=0.35)
        bottom = np.add(bottom, value1)
        bottom_compare = np.add(bottom_compare, value2)

        # ax.bar([xx-shift for xx in x], v if len(kernel_breakdown) != 0 else [0]*len(x), bottom=bottom, label=k, color=cmap[k], width=0.35)
        # ax.bar([xx+shift for xx in x], compare_breakdown[k] if len(compare_breakdown) != 0 else [0]*len(x), bottom=bottom_compare, color=cmap[k], width=0.35)

        # bottom = np.add(bottom, v)
        # bottom_compare = np.add(bottom_compare, compare_breakdown[k])

    ax.legend(fontsize=4, ncol=8, bbox_to_anchor=(0.45, 1.1), loc="upper center")
    xxlabel = [
        val
        for pair in zip(
            [xx - shift for xx in x[:num_separated_bar]],
            [xx + shift for xx in x[:num_separated_bar]],
        )
        for val in pair
    ] + list(x[num_separated_bar:])
    plt.xticks(
        xxlabel,
        ["P", "D"] * num_separated_bar + [""] * (len(x) - num_separated_bar),
        fontsize=6,
    )
    # ax.set_xticks(ax.get_xticks()[:8])

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(x, labels=["\n" + xl for xl in xlabel], fontsize=6)

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(
        [0, 2, 4, 6.5], labels=["\n\n" + xl for xl in secondary_xlabel], fontsize=6
    )
    # sec.tick_params(bottom = False)
    # sec.set_xticklabels([])
    sec.set(xlabel=None)
    sec.tick_params(bottom=False)  # remove the ticks

    sec2 = ax.secondary_xaxis(location=0)
    sec2.set_xticks([-0.5, 0.5, 3.5, 4.5, 8.5], labels=[])
    sec2.tick_params("x", length=25, width=0.8)
    ax.set_xlim(-0.5, 8.5)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-1]
        + " "
        + exp_name
        + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )
    wrapup_graph(
        plt,
        ax,
        "\n\n" + exp_name,
        xlabel,
        "Execution Time Breakdown (ms)",
        title_name,
        save_folder_path,
        "_decoding_step_operator_breakdown_overall.pdf",
        dpi,
        nested=nested,
        file_name_passed=file_name_passed,
    )


def graph_overall_compare_separate_ratio(
    kernel_breakdown,
    compare_breakdown,
    key_order,
    xlabel,
    secondary_xlabel,
    exp_name,
    save_folder_path,
    nested=False,
    file_name_passed=False,
):
    fig, ax, dpi = prep_graph()

    steps_len = len(xlabel)
    bottom = [0] * steps_len
    bottom_compare = [0] * steps_len
    bottom_real = [0] * steps_len
    bottom_compare_real = [0] * steps_len
    x = np.arange(steps_len)

    shift = 0.2

    num_separated_bar = 4
    total_time = list()
    total_time_compare = list()
    for i in range(steps_len):
        total_time.append(sum([v[i] for v in kernel_breakdown.values()]))
        total_time_compare.append(sum([v[i] for v in compare_breakdown.values()]))

    # for idx, (k, v) in enumerate(kernel_breakdown.items()):
    for idx, k in enumerate(key_order):
        label1 = [float(xx) for xx in x.copy()]
        label2 = [float(xx) for xx in x.copy()]

        value1 = list()
        value2 = list()
        value1_real = list()
        value2_real = list()
        print(k)
        for idxidx, vv in enumerate(kernel_breakdown[k]):
            print(xlabel[idxidx])
            if idxidx < num_separated_bar:
                value1.append(vv / total_time[idxidx] * 100)
                value2.append(
                    compare_breakdown[k][idxidx] / total_time_compare[idxidx] * 100
                )
                value1_real.append(vv)
                value2_real.append(compare_breakdown[k][idxidx])
                label1[idxidx] -= shift
                label2[idxidx] += shift
                print("Prefill: ", vv / total_time[idxidx] * 100)
                print(
                    "Decode: ",
                    compare_breakdown[k][idxidx] / total_time_compare[idxidx] * 100,
                )
                # print("Ratio: ", (vv+compare_breakdown[k][idxidx])/(total_time[idxidx]+total_time_compare[idxidx])*100)
            else:
                value1.append(0)
                value2.append(
                    vv / total_time[idxidx] * 100 if total_time[idxidx] > 0 else 0
                )
                value1_real.append(0)
                value2_real.append(vv)
                print(
                    "Ratio: ",
                    vv / total_time[idxidx] * 100 if total_time[idxidx] > 0 else 0,
                )

        ax.bar(label1, value1, bottom=bottom, label=k, color=cmap[k], width=0.35)
        ax.bar(label2, value2, bottom=bottom_compare, color=cmap[k], width=0.35)
        # ax.bar(label1, value1, bottom=bottom, label=k, width=0.35)
        # ax.bar(label2, value2, bottom=bottom_compare, width=0.35)
        bottom = np.add(bottom, value1)
        bottom_compare = np.add(bottom_compare, value2)
        bottom_real = np.add(bottom_real, value1_real)
        bottom_compare_real = np.add(bottom_compare_real, value2_real)

        # ax.bar([xx-shift for xx in x], v if len(kernel_breakdown) != 0 else [0]*len(x), bottom=bottom, label=k, color=cmap[k], width=0.35)
        # ax.bar([xx+shift for xx in x], compare_breakdown[k] if len(compare_breakdown) != 0 else [0]*len(x), bottom=bottom_compare, color=cmap[k], width=0.35)

        # bottom = np.add(bottom, v)
        # bottom_compare = np.add(bottom_compare, compare_breakdown[k])

    times = list()
    baseline = bottom_real[0]
    for idx, (a, b) in enumerate(zip(bottom_real, bottom_compare_real)):
        if idx < num_separated_bar:
            times.append(a / baseline)
            times.append(b / baseline)
        else:
            times.append(b / baseline)

    plt.ylim(0, 103)

    ax.legend(fontsize=6, ncol=4, bbox_to_anchor=(0.45, 1.35), loc="upper center")
    xxlabel = [
        val
        for pair in zip(
            [xx - shift for xx in x[:num_separated_bar]],
            [xx + shift for xx in x[:num_separated_bar]],
        )
        for val in pair
    ] + list(x[num_separated_bar:])
    plt.xticks(
        xxlabel,
        ["P", "D"] * num_separated_bar + [""] * (len(x) - num_separated_bar),
        fontsize=6,
    )
    # ax.set_xticks(ax.get_xticks()[:8])

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(x, labels=["\n" + xl for xl in xlabel], fontsize=6)

    # print(times)
    # # Set an offset that is used to bump the label up a bit above the bar.
    # y_offset = 102
    # # Add labels to each bar.
    # for i, t in enumerate(times):
    #     if i < num_separated_bar*2:
    #         if i==0:
    #             ax.text(i//2-shift, y_offset+6, f'{baseline:.1f}ms', ha='center', weight='bold', fontsize=4.5)
    #         elif i%2==1:
    #             ax.text(i//2+shift, y_offset, f'{times[i]:.1f}x', ha='center', weight='bold', fontsize=4.5)
    #         else:
    #             ax.text(i//2-shift, y_offset, f'{times[i]:.1f}x', ha='center', weight='bold', fontsize=4.5)

    #     else:
    #         ax.text(i-num_separated_bar, y_offset, f'{times[i]:.1f}x', ha='center', weight='bold', fontsize=4.5)

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(
        [0, 2, 5.5, 8], labels=["\n\n" + xl for xl in secondary_xlabel], fontsize=6
    )
    # sec.tick_params(bottom = False)
    # sec.set_xticklabels([])
    sec.set(xlabel=None)
    sec.tick_params(bottom=False)  # remove the ticks

    sec2 = ax.secondary_xaxis(location=0)
    sec2.set_xticks([0.5, 3.5, 7.5, 8.5], labels=[])
    sec2.tick_params("x", length=25, width=0.8)
    ax.set_xlim(-0.65, 8.5)

    major_ticks = np.arange(0, 101, 20)

    ax.set_yticks(major_ticks)

    # And a corresponding grid
    ax.grid(which="both")

    # Or if you want different settings for the grids:
    ax.grid(which="minor", alpha=0.2)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-1]
        + " "
        + exp_name
        + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )
    wrapup_graph(
        plt,
        ax,
        "\n\n" + exp_name,
        xlabel,
        "Execution Time Breakdown (%)",
        title_name,
        save_folder_path,
        "_decoding_step_operator_breakdown_overall.pdf",
        dpi,
        nested=nested,
        file_name_passed=file_name_passed,
    )


def graph_overall_ratio_compare(
    kernel_breakdown,
    compare_breakdown,
    xlabel,
    exp_name,
    save_folder_path,
    file_name_passed=False,
):
    fig, ax, dpi = prep_graph(nested=True)
    assert kernel_breakdown.keys() == compare_breakdown.keys()
    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [0] * steps_len
    bottom_compare = [0] * steps_len
    x = np.arange(steps_len)

    shift = 0.2

    total_time = list()
    total_time_compare = list()
    for i in range(steps_len):
        total_time.append(sum([v[i] for v in kernel_breakdown.values()]))
        total_time_compare.append(sum([v[i] for v in compare_breakdown.values()]))

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        ax.bar(
            [xx - shift for xx in x],
            (
                [vv / t * 100 if t != 0 else 0 for vv, t in zip(v, total_time)]
                if len(kernel_breakdown) != 0
                else [0] * len(x)
            ),
            bottom=bottom,
            label=k,
            color=cmap[k],
            width=0.35,
        )
        ax.bar(
            [xx + shift for xx in x],
            (
                [
                    vv / t * 100 if t != 0 else 0
                    for vv, t in zip(compare_breakdown[k], total_time_compare)
                ]
                if len(compare_breakdown) != 0
                else [0] * len(x)
            ),
            bottom=bottom_compare,
            color=cmap[k],
            width=0.35,
        )
        bottom = np.add(
            bottom, [vv / t * 100 if t != 0 else 0 for vv, t in zip(v, total_time)]
        )
        bottom_compare = np.add(
            bottom_compare,
            [
                vv / t * 100 if t != 0 else 0
                for vv, t in zip(compare_breakdown[k], total_time_compare)
            ],
        )

    ax.legend(fontsize=4)
    plt.xticks(
        [
            val
            for pair in zip([xx - shift for xx in x], [xx + shift for xx in x])
            for val in pair
        ],
        ["w/", "w/o"] * steps_len,
        fontsize=6,
    )

    sec = ax.secondary_xaxis(location=0)
    if len(xlabel) == 20:
        sec.set_xticks(
            x, labels=[xl + "        " for xl in xlabel], fontsize=6, rotation=90
        )
    else:
        sec.set_xticks(x, labels=["\n" + xl for xl in xlabel], fontsize=6)
    sec.tick_params(bottom=False)

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-1]
        + " "
        + exp_name
        + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )
    wrapup_graph(
        plt,
        ax,
        exp_name,
        xlabel,
        "Execution Time Breakdown (%)",
        title_name,
        save_folder_path,
        "_decoding_step_operator_breakdown_overall_ratio.pdf",
        dpi,
        nested=True,
        file_name_passed=file_name_passed,
    )


def graph_overall_grouped(
    kernel_breakdown, xlabel, exp_name, save_folder_path, secondary_xlabel
):
    fig, ax, dpi = prep_graph(nested=True)

    if args.simplify:
        new_kernel_breakdown = {"Compute": list(), "Communication": list()}
        new_list = list()
        for k, v in kernel_breakdown.items():
            if k == "Copy" or k == "Idle":
                continue
            elif k == "Communication":
                new_kernel_breakdown["Communication"] = kernel_breakdown[
                    "Communication"
                ]
            else:
                if len(new_list) == 0:
                    new_list = v
                else:
                    new_list = [a + b for a, b in zip(new_list, v)]
        new_kernel_breakdown["Compute"] = new_list
        kernel_breakdown = new_kernel_breakdown

    labels = list(kernel_breakdown.keys())

    steps_len = len(xlabel)
    # steps = np.arange(steps_len)
    bottom = [[0] * len(secondary_xlabel)] * steps_len
    x = np.arange(steps_len)

    from matplotlib import colormaps

    colormap = colormaps["Set3"].colors

    x = np.arange(len(secondary_xlabel))
    shift = 0.2

    for idx, (k, v) in enumerate(kernel_breakdown.items()):
        if k == "Copy":
            continue
        for idx2, bs in enumerate(xlabel):
            # ax.bar([xx-shift*(3-idx2) if idx2<=3 else xx+shift*(idx2-3) for xx in x], v[idx2::len(xlabel)] if len(kernel_breakdown) != 0 else [0]*len(x), bottom=bottom[idx2], label=k if idx2==0 else None, color=cmap[k], width=shift)
            ax.bar(
                [
                    xx - shift * (1 - idx2) if idx2 <= 1 else xx + shift * (idx2 - 1)
                    for xx in x
                ],
                v[idx2 :: len(xlabel)] if len(kernel_breakdown) != 0 else [0] * len(x),
                bottom=bottom[idx2],
                label=k if idx2 == 0 else None,
                color=cmap[k],
                width=shift,
            )
            bottom[idx2] = np.add(bottom[idx2], v[idx2 :: len(xlabel)])

    # plt.xticks([val for pair in zip([xx-shift*3 for xx in x], [xx-shift*2 for xx in x], [xx-shift*1 for xx in x], [xx-shift*0 for xx in x], [xx+shift*1 for xx in x], [xx+shift*2 for xx in x], [xx+shift*3 for xx in x]) for val in pair], xlabel*len(secondary_xlabel), fontsize=6)
    plt.xticks(
        [
            val
            for pair in zip(
                [xx - shift * 1 for xx in x],
                [xx - shift * 0 for xx in x],
                [xx + shift * 1 for xx in x],
                [xx + shift * 2 for xx in x],
            )
            for val in pair
        ],
        xlabel * len(secondary_xlabel),
        fontsize=6,
    )

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(
        x, labels=["\n\n" + sx for sx in secondary_xlabel], fontsize=FONTSIZE
    )  # , rotation=90)

    # label the classes:
    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks([1, 3, 6.5], labels=["\n\nMammals", "\n\nReptiles", "\n\nBirds"])
    sec.tick_params("x", length=0)

    # # lines between the classes:
    # sec2 = ax.secondary_xaxis(location=0)
    # sec2.set_xticks([-0.5, 2.5, 4.5, 8.5], labels=[])
    # sec2.tick_params('x', length=40, width=1.5)
    # ax.set_xlim(-0.6, 8.6)

    # Rotating X-axis labels
    ax.tick_params(axis="x")  # , rotation=90)
    ax.set_ylabel("End-to-End Inference Runtime (ms)")

    # ax.legend(loc="right", fontsize=6)
    ax.legend(loc="upper center", ncol=5, fontsize=FONTSIZE, bbox_to_anchor=(0.5, 1.18))

    folder_name_split = args.json_file.split("/")
    title_name = (
        folder_name_split[-1]
        + " "
        + re.sub("\n", "", exp_name)
        + " Exp\nOperator Breakdown w/ GPU Idle(Inference)\n"
        + (
            "Warmup with 5 examples, profile result for 6th inference sample"
            if "t2i" in args.json_file
            else "Warmup with 10 examples, profile result for 11th inference sample"
        )
    )
    wrapup_graph(
        plt,
        ax,
        exp_name,
        xlabel,
        "Execution Time Breakdown (ms)",
        title_name,
        save_folder_path,
        "_decoding_step_operator_breakdown_overall.pdf",
        dpi,
    )


def gather_result(profile_result, overall_breakdown, add_dummy=0, nested=False):
    if profile_result != dict():
        if len(overall_breakdown) == 0:
            for k in profile_result.keys():
                overall_breakdown[k] = [0] * add_dummy if add_dummy > 0 else list()

        assert profile_result.keys() == overall_breakdown.keys()
        for k, v in profile_result.items():
            if k in overall_breakdown:
                overall_breakdown[k].append(sum(v))
    else:
        if len(overall_breakdown) == 0:
            return add_dummy + 1
        else:
            for k, v in overall_breakdown.items():
                overall_breakdown[k].append(0)
    return 0


def gather_result_separate(
    profile_result, overall_breakdown, add_dummy=0, nested=False, merge_sample=False
):
    if profile_result != dict():
        if len(overall_breakdown) == 0:
            for k in profile_result.keys():
                overall_breakdown[k] = list()

        # assert profile_result.keys() == overall_breakdown.keys()
        for k, v in profile_result.items():
            if k in overall_breakdown:
                if merge_sample:
                    if len(overall_breakdown[k]) == 0:
                        overall_breakdown[k] = v
                    else:
                        overall_breakdown[k] += profile_result[k]
                else:
                    overall_breakdown[k].append(v)
            else:
                overall_breakdown[k] = v
    # else:
    #     if len(overall_breakdown)==0:
    #         return add_dummy+1
    #     else:
    #         for k, v in overall_breakdown.items():
    #             overall_breakdown[k].append(0)
    # return 0


if args.json_file.split("/")[-1] == "":
    print('Please remove "/" at the end of --json-file argument')
    exit(0)

if args.batch_size and not args.multigpu:
    assert "retrieval" not in args.json_file
    # Batch Iterate
    BATCH_SIZE = ["1", "4", "8", "16", "32"]
    overall_breakdown = dict()
    for bs in BATCH_SIZE:
        exp_info = args.json_file.split("/")[-1].split("bs")
        file_path = (
            "/".join(args.json_file.split("/")[:-1])
            + "/"
            + exp_info[0]
            + "bs"
            + bs
            + "."
            + ".".join(exp_info[1].split(".")[1:])
            + "/profile.json"
        )
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, overall_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = (
        args.graph_path
        + "/"
        + ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "")
        + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")
        + exp_info[0]
        + ".".join(exp_info[1].split(".")[1:])
        + "/"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall(overall_breakdown, BATCH_SIZE, "batch_size", save_folder_path)
    graph_overall_ratio(overall_breakdown, BATCH_SIZE, "batch_size", save_folder_path)

elif args.n_retrieved_doc:
    assert "retrieval" in args.json_file

    # NRETRIEVED_DOCS Iterate
    NRETRIEVED_DOCS = ["1", "2", "3", "4"]
    overall_breakdown = dict()
    for nd in NRETRIEVED_DOCS:
        file_path = args.json_file + ".n_retrieved_docs" + nd + "/profile.json"
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, overall_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = (
        args.graph_path
        + "/"
        + ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "")
        + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")
        + exp_info[0]
        + ".".join(exp_info[1].split(".")[1:])
        + "/"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall(
        overall_breakdown, NRETRIEVED_DOCS, "n_retrieved_docs", save_folder_path
    )
    graph_overall_ratio(
        overall_breakdown, NRETRIEVED_DOCS, "n_retrieved_docs", save_folder_path
    )

elif args.both and not args.compare_efficient_attn:
    assert "retrieval" in args.json_file

    # BATCH_SIZE and NRETRIEVED_DOCS Iterate
    BATCH_SIZE = ["bs1", "bs4", "bs8", "bs16"]  # , "bs32"]
    NRETRIEVED_DOCS = [
        "n_retrieved_docs1",
        "n_retrieved_docs2",
        "n_retrieved_docs3",
        "n_retrieved_docs4",
    ]
    overall_breakdown = dict()
    for bs in BATCH_SIZE:
        for nd in NRETRIEVED_DOCS:
            exp_info = args.json_file.split("/")[-1].split("bs")
            file_path = (
                "/".join(args.json_file.split("/")[:-1])
                + "/"
                + exp_info[0]
                + bs
                + "."
                + ".".join(exp_info[1].split(".")[1:])
                + "."
                + nd
                + "/profile.json"
            )
            profile_result = parse_file(file_path, plot_graph=False)
            gather_result(profile_result, overall_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = (
        args.graph_path
        + "/"
        + ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "")
        + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")
        + exp_info[0]
        + ".".join(exp_info[1].split(".")[1:])
        + "/"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall(
        overall_breakdown,
        [".".join(ip) for ip in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))],
        "batch_size&n_retrieved_docs",
        save_folder_path,
    )
    graph_overall_ratio(
        overall_breakdown,
        [".".join(ip) for ip in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))],
        "batch_size&n_retrieved_docs",
        save_folder_path,
    )

elif args.compare_efficient_attn and not args.both:
    assert args.compare_dir != ""
    # Batch Iterate
    BATCH_SIZE = ["1", "4", "8", "16", "32"]
    overall_breakdown = dict()
    compare_breakdown = dict()
    for bs in BATCH_SIZE:
        exp_info = args.json_file.split("/")[-1].split("bs")
        file_path = (
            "/".join(args.json_file.split("/")[:-1])
            + "/"
            + exp_info[0]
            + "bs"
            + bs
            + "."
            + ".".join(exp_info[1].split(".")[1:])
            + "/profile.json"
        )
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, overall_breakdown)

        file_path = (
            args.compare_dir
            + "/"
            + exp_info[0]
            + "bs"
            + bs
            + "."
            + ".".join(exp_info[1].split(".")[1:])
            + "/profile.json"
        )
        profile_result = parse_file(file_path, plot_graph=False)
        gather_result(profile_result, compare_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = (
        args.graph_path
        + "/wo_efficient_attn/compare/"
        + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")
        + exp_info[0]
        + ".".join(exp_info[1].split(".")[1:])
        + "/"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall_compare(
        overall_breakdown, compare_breakdown, BATCH_SIZE, "batch_size", save_folder_path
    )
    graph_overall_ratio_compare(
        overall_breakdown, compare_breakdown, BATCH_SIZE, "batch_size", save_folder_path
    )

elif args.compare_efficient_attn and args.both:
    assert args.compare_dir != ""
    assert "retrieval" in args.json_file

    # Batch Iterate
    BATCH_SIZE = ["bs1", "bs4", "bs8", "bs16"]  # , "bs32"]
    NRETRIEVED_DOCS = [
        "original",
        "n_retrieved_docs1",
        "n_retrieved_docs2",
        "n_retrieved_docs3",
        "n_retrieved_docs4",
    ]

    overall_breakdown = dict()
    compare_breakdown = dict()
    for bs in BATCH_SIZE:
        for nd in NRETRIEVED_DOCS:
            if nd == "original":
                original_path = re.sub(
                    "default_retrieval_template.",
                    "cm3v2_template." if "t2i" not in args.json_file else "",
                    re.sub(
                        "flamingo_retrieval_v2_template.",
                        "cm3v2_template." if "t2i" not in args.json_file else "",
                        args.json_file,
                    ),
                )
                exp_info = original_path.split("/")[-1].split("bs")
                file_path = (
                    "/".join(original_path.split("/")[:-2])
                    + "/"
                    + exp_info[0]
                    + bs
                    + "."
                    + ".".join(exp_info[1].split(".")[1:])
                    + "/profile.json"
                )
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, overall_breakdown)

                file_path = (
                    re.sub(
                        r"retrieval",
                        "",
                        re.sub(
                            r"flamingo_retrieval_v2_template.",
                            "cm3v2_template." if "t2i" not in args.json_file else "",
                            args.compare_dir,
                        ),
                    )
                    + "/"
                    + exp_info[0]
                    + bs
                    + "."
                    + ".".join(exp_info[1].split(".")[1:])
                    + "/profile.json"
                )
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, compare_breakdown)

            else:
                exp_info = args.json_file.split("/")[-1].split("bs")
                file_path = (
                    "/".join(args.json_file.split("/")[:-1])
                    + "/"
                    + exp_info[0]
                    + bs
                    + "."
                    + ".".join(exp_info[1].split(".")[1:])
                    + "."
                    + nd
                    + "/profile.json"
                )
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, overall_breakdown)

                file_path = (
                    args.compare_dir
                    + "/"
                    + exp_info[0]
                    + bs
                    + "."
                    + ".".join(exp_info[1].split(".")[1:])
                    + "."
                    + nd
                    + "/profile.json"
                )
                profile_result = parse_file(file_path, plot_graph=False)
                gather_result(profile_result, compare_breakdown)

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split("bs")
    save_folder_path = (
        args.graph_path
        + "/wo_efficient_attn/compare/"
        + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")
        + exp_info[0]
        + ".".join(exp_info[1].split(".")[1:])
        + "/"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    graph_overall_compare(
        overall_breakdown,
        compare_breakdown,
        [".".join(l) for l in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))],
        "batch_size",
        save_folder_path,
    )
    graph_overall_ratio_compare(
        overall_breakdown,
        compare_breakdown,
        [".".join(l) for l in list(itertools.product(BATCH_SIZE, NRETRIEVED_DOCS))],
        "batch_size",
        save_folder_path,
    )


elif args.multigpu and args.batch_size:
    # NGPU_NNODE=[("1-1"), ("4-1"), ("8-1")]#, ("4-2"), ("2-4"), ("1-8")]
    # BATCH_SIZE=["1", "4", "8", "16", "32", "64", "128"]
    NGPU_NNODE = [("1-1"), ("2-1"), ("4-1"), ("8-1")]
    BATCH_SIZE = ["1", "4", "8", "16", "32", "64", "128"]

    overall_breakdown = dict()
    overall_latency_breakdown = dict()
    warmup = 15
    num_sample = 3 if "txt_to_img" in args.json_file else 5
    add_dummy = 0
    add_dummy2 = 0
    for bs in BATCH_SIZE:
        for config in NGPU_NNODE:
            ngpu, nnode = config.split("-")
            file_dir = re.sub(
                r"[0-9]gpu_[0-9]node", ngpu + "gpu_" + nnode + "node", args.json_file
            )
            multigpu_breakdown = dict()
            multigpu_latency_breakdown = dict()
            all_exist = True

            for sample_id in range(warmup, warmup + num_sample):
                sample_multigpu_breakdown = dict()
                sample_multigpu_latency_breakdown = dict()

                for g in range(int(ngpu) * int(nnode)):
                    if "txt_to_img" in args.json_file:
                        _sample_multigpu_breakdown = dict()
                        _sample_multigpu_latency_breakdown = dict()
                        for i in range(128, 1024, 64):
                            exp_info = file_dir.split("/")[-1].split("ncs.")
                            file_path = (
                                "/".join(file_dir.split("/")[:-1])
                                + "/"
                                + exp_info[0]
                                + "ncs."
                                + bs
                                + "."
                                + ".".join(exp_info[1].split(".")[1:])
                                + "/profile_sample_"
                                + str(sample_id)
                                + "_"
                                + str(i)
                                + "_gpu_"
                                + str(g)
                                + ".json"
                            )
                            if not os.path.isfile(file_path):
                                print(file_path)
                            profile_result = parse_file(file_path, plot_graph=False)
                            # profile_result = {'MODULE_PREPROC_ENCODE_IMAGES_AG': [19.204000000000004, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Misc': [8.792000000000002, 0.43700000000000033, 0.4300000000000003, 0.4280000000000003, 0.4240000000000003, 0.4240000000000003, 0.4270000000000003, 0.4310000000000003, 0.4300000000000003, 0.4260000000000003], 'MODULE_ParallelEmbedding_AG': [0.092, 0.003, 0.003, 0.003, 0.004, 0.004, 0.003, 0.004, 0.004, 0.003], 'MODULE_FusedRMSNorm_AG': [4.769000000000001, 3.2909999999999955, 3.2749999999999955, 3.258999999999997, 3.2539999999999956, 3.254999999999996, 3.2509999999999972, 3.251999999999996, 3.2459999999999964, 3.258999999999996], 'MODULE_LayerNorm_AG': [8.097999999999997, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024], 'MODULE__InnerAttention_AG': [25.89500000000002, 6.263999999999969, 6.218999999999967, 6.200999999999968, 6.17399999999997, 6.200999999999968, 6.207999999999968, 6.209999999999968, 6.206999999999969, 6.2039999999999695], 'Copy': [0.35, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004], 'MODULE_SCORING_AG': [0.06300000000000001, 0.08200000000000002, 0.08400000000000002, 0.08200000000000002, 0.08300000000000002, 0.08300000000000002, 0.08400000000000003, 0.08400000000000002, 0.08100000000000002, 0.08500000000000002], 'Communication': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Linear': [314.0490000000001, 40.52999999999994, 40.47799999999992, 40.448999999999934, 40.46899999999994, 40.45599999999993, 40.43899999999994, 40.48199999999996, 40.44099999999993, 40.44599999999994], 'Idle': [13.661999999999864, 78.36300000000011, 77.53000000000011, 80.00700000000012, 76.6300000000001, 80.6880000000001, 76.9760000000001, 80.09600000000007, 77.9250000000001, 80.33800000000008]}
                            if len(profile_result) == 0:
                                all_exist = False
                            gather_result(profile_result, _sample_multigpu_breakdown)
                            gather_result(
                                {
                                    k: [vv / int(bs) for vv in v]
                                    for k, v in profile_result.items()
                                },
                                _sample_multigpu_latency_breakdown,
                            )

                        if all_exist:
                            _sample_multigpu_breakdown = {
                                k: [np.sum(v)]
                                for k, v in _sample_multigpu_breakdown.items()
                            }
                            _sample_multigpu_latency_breakdown = {
                                k: [np.sum(v)]
                                for k, v in _sample_multigpu_latency_breakdown.items()
                            }
                        else:
                            _sample_multigpu_breakdown = dict()
                            _sample_multigpu_latency_breakdown = dict()

                        gather_result(
                            _sample_multigpu_breakdown, sample_multigpu_breakdown
                        )
                        gather_result(
                            _sample_multigpu_latency_breakdown,
                            sample_multigpu_latency_breakdown,
                        )

                    else:
                        exp_info = file_dir.split("/")[-1].split("mbs.")
                        file_path = (
                            "/".join(file_dir.split("/")[:-1])
                            + "/"
                            + exp_info[0]
                            + "mbs."
                            + bs
                            + "."
                            + ".".join(exp_info[1].split(".")[1:])
                            + "/profile_sample_"
                            + str(sample_id)
                            + "_gpu_"
                            + str(g)
                            + ".json"
                        )
                        if not os.path.isfile(file_path):
                            print(file_path)

                        profile_result = parse_file(file_path, plot_graph=False)

                        # profile_result = {'MODULE_PREPROC_ENCODE_IMAGES_AG': [19.204000000000004, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Misc': [8.792000000000002, 0.43700000000000033, 0.4300000000000003, 0.4280000000000003, 0.4240000000000003, 0.4240000000000003, 0.4270000000000003, 0.4310000000000003, 0.4300000000000003, 0.4260000000000003], 'MODULE_ParallelEmbedding_AG': [0.092, 0.003, 0.003, 0.003, 0.004, 0.004, 0.003, 0.004, 0.004, 0.003], 'MODULE_FusedRMSNorm_AG': [4.769000000000001, 3.2909999999999955, 3.2749999999999955, 3.258999999999997, 3.2539999999999956, 3.254999999999996, 3.2509999999999972, 3.251999999999996, 3.2459999999999964, 3.258999999999996], 'MODULE_LayerNorm_AG': [8.097999999999997, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024, 0.33600000000000024], 'MODULE__InnerAttention_AG': [25.89500000000002, 6.263999999999969, 6.218999999999967, 6.200999999999968, 6.17399999999997, 6.200999999999968, 6.207999999999968, 6.209999999999968, 6.206999999999969, 6.2039999999999695], 'Copy': [0.35, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004], 'MODULE_SCORING_AG': [0.06300000000000001, 0.08200000000000002, 0.08400000000000002, 0.08200000000000002, 0.08300000000000002, 0.08300000000000002, 0.08400000000000003, 0.08400000000000002, 0.08100000000000002, 0.08500000000000002], 'Communication': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Linear': [314.0490000000001, 40.52999999999994, 40.47799999999992, 40.448999999999934, 40.46899999999994, 40.45599999999993, 40.43899999999994, 40.48199999999996, 40.44099999999993, 40.44599999999994], 'Idle': [13.661999999999864, 78.36300000000011, 77.53000000000011, 80.00700000000012, 76.6300000000001, 80.6880000000001, 76.9760000000001, 80.09600000000007, 77.9250000000001, 80.33800000000008]}
                        # new_profile_result = dict()
                        # for k, v  in profile_result.items():
                        #     new_profile_result[re.sub("POSTPROC_GENERATE_TEXT", "(Postprocessing) Generate Text", (re.sub("PREPROC_ENCODE_IMAGES", "(Preprocessing) Encode Image", re.sub("SCORING", "Scoring", re.sub("_AG", "", re.sub("MODULE_", "", k))))))] = v
                        # del profile_result
                        # profile_result = new_profile_result

                        if len(profile_result) == 0:
                            all_exist = False
                        gather_result(profile_result, sample_multigpu_breakdown)
                        gather_result(
                            {
                                k: [vv / int(bs) for vv in v]
                                for k, v in profile_result.items()
                            },
                            sample_multigpu_latency_breakdown,
                        )

                for k, v in sample_multigpu_breakdown.items():
                    assert len(v) == int(ngpu) * int(nnode)
                    assert len(sample_multigpu_latency_breakdown[k]) == int(ngpu) * int(
                        nnode
                    )

                if all_exist:
                    sample_multigpu_breakdown = {
                        k: [np.average(v)] for k, v in sample_multigpu_breakdown.items()
                    }
                    sample_multigpu_latency_breakdown = {
                        k: [np.average(v)]
                        for k, v in sample_multigpu_latency_breakdown.items()
                    }
                else:
                    sample_multigpu_breakdown = dict()
                    sample_multigpu_latency_breakdown = dict()

                for k, v in sample_multigpu_breakdown.items():
                    assert len(v) == 1, len(v)
                for k, v in sample_multigpu_latency_breakdown.items():
                    assert len(v) == 1, len(v)

                gather_result(sample_multigpu_breakdown, multigpu_breakdown)
                gather_result(
                    sample_multigpu_latency_breakdown, multigpu_latency_breakdown
                )

            for k, v in multigpu_breakdown.items():
                assert len(v) == num_sample, len(v)
            for k, v in multigpu_latency_breakdown.items():
                assert len(v) == num_sample, len(v)

            multigpu_breakdown = {
                k: [np.average(v)] for k, v in multigpu_breakdown.items()
            }
            multigpu_latency_breakdown = {
                k: [np.average(v)] for k, v in multigpu_latency_breakdown.items()
            }

            add_dummy = gather_result(multigpu_breakdown, overall_breakdown, add_dummy)
            add_dummy2 = gather_result(
                multigpu_latency_breakdown, overall_latency_breakdown, add_dummy2
            )

    folder_name_split = args.json_file.split("/")
    exp_info = folder_name_split[-1].split(
        "ncs" if "txt_to_img" in args.json_file else "mbs"
    )
    save_folder_path = (
        args.graph_path
        + "/"
        + ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "")
        + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")
        + ("simplify/" if args.simplify else "")
        + exp_info[0]
        + ".".join(exp_info[1].split(".")[1:])
        + "/"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    # graph_overall_grouped(overall_breakdown, BATCH_SIZE, "\n\nbatch_size", save_folder_path, secondary_xlabel=NGPU_NNODE)
    graph_overall_grouped(
        overall_breakdown,
        NGPU_NNODE,
        "\n\nBatch size",
        save_folder_path,
        secondary_xlabel=BATCH_SIZE,
    )
    save_folder_path = (
        args.graph_path
        + "/"
        + ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "")
        + ("retrieval/" if "retrieval" in args.json_file else "batch_size_overall/")
        + ("simplify/" if args.simplify else "")
        + exp_info[0]
        + ".".join(exp_info[1].split(".")[1:])
        + "/latency/"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    # graph_overall_grouped(overall_latency_breakdown, BATCH_SIZE, "\n\nbatch_size", save_folder_path, secondary_xlabel=NGPU_NNODE)
    graph_overall_grouped(
        overall_latency_breakdown,
        NGPU_NNODE,
        "\n\nBatch size",
        save_folder_path,
        secondary_xlabel=BATCH_SIZE,
    )
elif args.export:
    file_paths = list()
    desired_prefixes_list = list()
    batch_size_list = [1, 4, 8, 16, 32, 64, 128]
    # Chameleon-34B (ImgTxt2Txt)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs."
            + str(bs)
            + ".umca.True.gm.text.ev.False/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs."
            + str(bs)
            + ".umca.True.gm.text.ev.False/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs."
            + str(bs)
            + ".umca.True.gm.text.ev.False/"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
    # Chameleon-34B (Img2Txt)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs."
            + str(bs)
            + ".umca.True.gm.text.ev.False/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs."
            + str(bs)
            + ".umca.True.gm.text.ev.False/"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
    # Chameleon-34B (Txt2Img)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."
            + str(bs)
            + ".en.image_gen.g.True/%j/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."
            + str(bs)
            + ".en.image_gen.g.True/%j/"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POST_PROC_IMAGE_DECODE_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POST_PROC_IMAGE_DECODE_AG"
        )
    # Chameleon-7B (ImgTxt2Txt)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown/it2t.cm3v21_109m_sft.bs"
            + str(bs)
            + ".textvqa.0_shot.cm3v2_template/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown/it2t.cm3v21_109m_sft.bs"
            + str(bs)
            + ".okvqa.0_shot.cm3v2_template/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown/it2t.cm3v21_109m_sft.bs"
            + str(bs)
            + ".vizwiz.0_shot.cm3v2_template/"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
    # Chameleon-7B (Img2Txt)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown/i2t.cm3v21_109m_sft.bs"
            + str(bs)
            + ".coco.0_shot.cm3v2_template/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown/i2t.cm3v21_109m_sft.bs"
            + str(bs)
            + ".flickr30k.0_shot.cm3v2_template/"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG"
        )
    # Chameleon-7B (Txt2Img)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown/t2i.cm3v21_109m_sft.bs"
            + str(bs)
            + ".coco_image.0_shot.cfg6.temp1.0.topp0.9.seed.1/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/cm3v2_breakdown/t2i.cm3v21_109m_sft.bs"
            + str(bs)
            + ".partiprompts.0_shot.cfg6.temp1.0.topp0.9.seed.1/"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POST_PROC_IMAGE_DECODE_AG"
        )
        desired_prefixes_list.append(
            "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POST_PROC_IMAGE_DECODE_AG"
        )

    # Codellama - 34B
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/bigcode_eval_34B_breakdown/1gpu_1node/HumanEval/batch_size_"
            + str(bs)
            + "/"
        )
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/bigcode_eval_34B_breakdown/1gpu_1node/MBPP/batch_size_"
            + str(bs)
            + "/"
        )
        desired_prefixes_list.append(
            "MODULE_Embedding_AG*MODULE_LlamaRMSNorm_AG*MODULE_Linear_AG*MODULE_LlamaRotaryEmbedding_AG*MODULE_SiLU_AG*MODULE_TEXT_DECODE_AG*MODULE_Attention_AG*MODULE_SCORING_AG"
        )
        desired_prefixes_list.append(
            "MODULE_Embedding_AG*MODULE_LlamaRMSNorm_AG*MODULE_Linear_AG*MODULE_LlamaRotaryEmbedding_AG*MODULE_SiLU_AG*MODULE_TEXT_DECODE_AG*MODULE_Attention_AG*MODULE_SCORING_AG"
        )

    batch_size_list = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    # HSTU - Triton
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/profile_results/batch_size_"
            + str(bs)
            + "/"
        )
        desired_prefixes_list.append(
            "MODULE_Embedding_AG*MODULE_Sigmoid_AG*MODULE_LayerNorm_AG*MODULE_Linear_AG*MODULE_Attention_AG"
        )

    # HSTU - Pytorch
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/profile_results/pytorch/batch_size_"
            + str(bs)
            + "/"
        )
        desired_prefixes_list.append(
            "MODULE_Embedding_AG*MODULE_Sigmoid_AG*MODULE_LayerNorm_AG*MODULE_Linear_AG*MODULE_Attention_AG"
        )

    batch_size_list = [1, 4, 8, 16, 32, 64, 128, 256, 384]
    # Seamless (S2TT)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/S2TT/batch_size_"
            + str(bs)
            + "/"
        )
        desired_prefixes_list.append(
            "MODULE_TorchSDPA_AG*MODULE_GLU_AG*MODULE_SiLU_AG*MODULE_Wav2Vec2FbankFeatureExtractor_AG*MODULE_KV_Cache_Reorder_AG*MODULE_ReLU_AG*MODULE_Conv1d_AG*MODULE_Linear_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_Dropout_AG*MODULE_StandardEmbedding_AG*MODULE_StandardLayerNorm_AG*MODULE_TiedProjection_AG"
        )
    # Seamless (S2ST)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/S2ST/batch_size_"
            + str(bs)
            + "/"
        )
        desired_prefixes_list.append(
            "MODULE_StandardLayerNorm_AG*MODULE_SiLU_AG*MODULE_Dropout_AG*MODULE_TiedProjection_AG*MODULE_Conv1d_AG*MODULE_KV_Cache_Reorder_AG*MODULE_Wav2Vec2FbankFeatureExtractor_AG*MODULE_ConvTranspose1d_AG*MODULE_GLU_AG*MODULE_Embedding_AG*MODULE_ReLU_AG*MODULE_Masked_Select_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_HardUpsampling_AG*MODULE_TorchSDPA_AG*MODULE_StandardEmbedding_AG*MODULE_Linear_AG"
        )
    # Seamless (T2TT)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/T2TT/batch_size_"
            + str(bs)
            + "/"
        )
        desired_prefixes_list.append(
            "MODULE_Dropout_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_TiedProjection_AG*MODULE_ReLU_AG*MODULE_KV_Cache_Reorder_AG*MODULE_Linear_AG*MODULE_TorchSDPA_AG*MODULE_StandardLayerNorm_AG*MODULE_StandardEmbedding_AG"
        )
    # Seamless (T2ST)
    for bs in batch_size_list:
        file_paths.append(
            "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/T2ST/batch_size_"
            + str(bs)
            + "/"
        )
        desired_prefixes_list.append(
            "MODULE_HardUpsampling_AG*MODULE_StandardLayerNorm_AG*MODULE_StandardEmbedding_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_TiedProjection_AG*MODULE_KV_Cache_Reorder_AG*MODULE_ConvTranspose1d_AG*MODULE_Embedding_AG*MODULE_ReLU_AG*MODULE_Linear_AG*MODULE_Masked_Select_AG*MODULE_Dropout_AG*MODULE_TorchSDPA_AG*MODULE_Conv1d_AG"
        )

    for idx, (fp, dp) in tqdm(enumerate(zip(file_paths, desired_prefixes_list))):
        desired_prefixes = list(set(dp.split("*")))
        sample_breakdown = dict()

        warmup = (
            15
            if len([path for path in glob.glob(fp + "/profile_sample_15*")]) > 0
            else (
                0
                if "hstu" in fp
                or fp
                == "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
                else 1
            )
        )
        num_sample = (
            3
            if "txt_to_img" in fp
            else len([path for path in glob.glob(fp + "/profile*.json")])
        )

        if num_sample == 0:
            print("Folder doesn't exists!!!! ", fp)
        for sample_id in range(warmup, warmup + num_sample):
            if "txt_to_img" in fp:
                _sample_breakdown = dict()

                for i in range(64, 1088, 64):
                    profile_result = parse_file(
                        fp
                        + "profile_sample_"
                        + str(sample_id)
                        + "_"
                        + str(i)
                        + "_gpu_0.json",
                        plot_graph=False,
                    )

                profile_result = parse_file(
                    fp
                    + (
                        "image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                        if "coco_image" in fp
                        else "image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                    )
                    + "profile_sample_"
                    + str(sample_id)
                    + "_last_gpu_0.json",
                    plot_graph=False,
                )

            else:
                if "cm3v2_breakdown/" in fp:
                    profile_result = parse_file(fp + "profile.json", plot_graph=False)
                else:
                    profile_result = parse_file(
                        fp + "profile_sample_" + str(sample_id) + "_gpu_0.json",
                        plot_graph=False,
                    )


elif args.figure1:
    model_name = [
        "IT-T\nCM3",
        "I-T\nCM3",
        "T-I\nCM3",
        "T-T\nCodeLlama",
        "HSTU",
        "S2TT\nSeamless",
        "S2ST\nSeamless",
        "T2TT\nSeamless",
        "T2ST\nSeamless",
    ]
    file_paths = [
        # Chameleon (ImgTxt2Txt)
        "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.16.umca.True.gm.text.ev.False/",
        # Chameleon (Img2Txt)
        "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.16.umca.True.gm.text.ev.False/",
        # Chameleon (Txt2Img)
        "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.16.en.image_gen.g.True/%j/",
        # Codellama
        "/fsx-atom/yejinlee/paper_submission_results/bigcode_eval_34B_breakdown/1gpu_1node/MBPP/batch_size_4/",
        # HSTU
        "/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/profile_results/pytorch/batch_size_32/",
        # Seamless (S2TT)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/S2TT/batch_size_128/",
        # Seamless (S2ST)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/S2ST/batch_size_128/",
        # Seamless (T2TT)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/T2TT/batch_size_384/",
        # Seamless (T2ST)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/T2ST/batch_size_384/",
    ]
    desired_prefixes_list = [
        # Chameleon (ImgTxt2Txt)
        "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG",
        # Chameleon (Img2Txt)
        "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG",
        # Chameleon (Txt2Img)
        "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POST_PROC_IMAGE_DECODE_AG",
        # Codellama
        "MODULE_Embedding_AG*MODULE_LlamaRMSNorm_AG*MODULE_Linear_AG*MODULE_LlamaRotaryEmbedding_AG*MODULE_SiLU_AG*MODULE_TEXT_DECODE_AG*MODULE_Attention_AG",
        # HSTU
        "MODULE_Embedding_AG*MODULE_Sigmoid_AG*MODULE_LayerNorm_AG*MODULE_Linear_AG*MODULE_Attention_AG",
        # # Seamless (S2TT)
        "MODULE_TorchSDPA_AG*MODULE_GLU_AG*MODULE_SiLU_AG*MODULE_Wav2Vec2FbankFeatureExtractor_AG*MODULE_KV_Cache_Reorder_AG*MODULE_ReLU_AG*MODULE_Conv1d_AG*MODULE_Linear_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_Dropout_AG*MODULE_StandardEmbedding_AG*MODULE_StandardLayerNorm_AG*MODULE_TiedProjection_AG",
        # Seamless (S2ST)
        "MODULE_StandardLayerNorm_AG*MODULE_SiLU_AG*MODULE_Dropout_AG*MODULE_TiedProjection_AG*MODULE_Conv1d_AG*MODULE_KV_Cache_Reorder_AG*MODULE_Wav2Vec2FbankFeatureExtractor_AG*MODULE_ConvTranspose1d_AG*MODULE_GLU_AG*MODULE_Embedding_AG*MODULE_ReLU_AG*MODULE_Masked_Select_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_HardUpsampling_AG*MODULE_TorchSDPA_AG*MODULE_StandardEmbedding_AG*MODULE_Linear_AG",
        # Seamless (T2TT)
        "MODULE_Dropout_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_TiedProjection_AG*MODULE_ReLU_AG*MODULE_KV_Cache_Reorder_AG*MODULE_Linear_AG*MODULE_TorchSDPA_AG*MODULE_StandardLayerNorm_AG*MODULE_StandardEmbedding_AG",
        # Seamless (T2ST)
        "MODULE_HardUpsampling_AG*MODULE_StandardLayerNorm_AG*MODULE_StandardEmbedding_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_TiedProjection_AG*MODULE_KV_Cache_Reorder_AG*MODULE_ConvTranspose1d_AG*MODULE_Embedding_AG*MODULE_ReLU_AG*MODULE_Linear_AG*MODULE_Masked_Select_AG*MODULE_Dropout_AG*MODULE_TorchSDPA_AG*MODULE_Conv1d_AG",
    ]

    dp_collection = set()
    overall_breakdown = dict()

    for k in cmap.keys():
        overall_breakdown[k] = list()

    for idx, (fp, dp) in enumerate(zip(file_paths, desired_prefixes_list)):
        desired_prefixes = list(set(dp.split("*")))
        sample_breakdown = dict()

        # warmup=15 if len([path for path in glob.glob(fp+"/profile_sample_15*")]) > 0 else 1
        warmup = (
            15
            if len([path for path in glob.glob(fp + "/profile_sample_15*")]) > 0
            else (
                0
                if "hstu" in fp
                or fp
                == "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
                else 1
            )
        )
        # num_sample=3 if "txt_to_img" in fp or "" else 5
        num_sample = (
            3
            if "txt_to_img" in fp
            else len([path for path in glob.glob(fp + "/profile_sample*")])
        )

        for sample_id in range(warmup, warmup + num_sample):
            if "txt_to_img" in fp:
                _sample_breakdown = dict()

                for i in range(64, 1088, 64):
                    profile_result = parse_file(
                        fp
                        + "profile_sample_"
                        + str(sample_id)
                        + "_"
                        + str(i)
                        + "_gpu_0.json",
                        plot_graph=False,
                    )
                    gather_result(profile_result, _sample_breakdown)

                profile_result = parse_file(
                    fp
                    + (
                        "image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                        if "coco_image" in fp
                        else "image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                    )
                    + "profile_sample_"
                    + str(sample_id)
                    + "_last_gpu_0.json",
                    plot_graph=False,
                )
                gather_result(profile_result, _sample_breakdown)

                _sample_breakdown = {
                    k: [np.sum(v)] for k, v in _sample_breakdown.items()
                }
                gather_result(_sample_breakdown, sample_breakdown)

            else:
                profile_result = parse_file(
                    fp + "profile_sample_" + str(sample_id) + "_gpu_0.json",
                    plot_graph=False,
                )
                # profile_result = {'(Preprocessing) Encode Image': [15.238000000000008, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Misc': [8.823000000000006, 0.43800000000000033, 0.4280000000000003, 0.43200000000000033, 0.4270000000000003, 0.4230000000000003, 0.4230000000000003, 0.4250000000000003, 0.43300000000000033, 0.43400000000000033], 'LayerNorm': [12.997999999999998, 3.6359999999999952, 3.5939999999999954, 3.5929999999999964, 3.5909999999999953, 3.5889999999999964, 3.584999999999997, 3.5929999999999964, 3.586999999999996, 3.5989999999999958], 'Copy': [0.352, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004], 'Scoring': [0.06300000000000001, 0.08300000000000002, 0.08200000000000002, 0.08100000000000002, 0.08300000000000003, 0.08200000000000002, 0.08100000000000002, 0.08500000000000002, 0.08300000000000002, 0.08500000000000002], 'Linear': [315.8960000000001, 40.487999999999936, 40.42199999999994, 40.40599999999994, 40.48999999999995, 40.45299999999991, 40.40899999999992, 40.46499999999993, 40.48299999999995, 40.487999999999914], 'Embedding': [0.092, 0.004, 0.003, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003], 'Attention': [26.062999999999988, 6.31099999999998, 6.221999999999979, 6.220999999999983, 6.248999999999979, 6.246999999999986, 6.252999999999983, 6.2619999999999845, 6.259999999999983, 6.255999999999982], 'Idle': [-9.727000000000032, 97.79900000000009, 92.46100000000007, 103.96900000000008, 90.72800000000007, 83.71700000000011, 107.91000000000008, 97.30100000000007, 98.40400000000008, 106.65400000000011]}
                gather_result(profile_result, sample_breakdown)

        sample_breakdown = {k: [np.average(v)] for k, v in sample_breakdown.items()}

        dp_collection = dp_collection.union(set(sample_breakdown.keys()))

        for k, v in overall_breakdown.items():
            if k in sample_breakdown:
                overall_breakdown[k].append(sum(sample_breakdown[k]))
            else:
                overall_breakdown[k].append(0)

    del_keys = list()
    for k, v in overall_breakdown.items():
        if sum(v) == 0:
            del_keys.append(k)
    for k in del_keys:
        del overall_breakdown[k]
    print(overall_breakdown)

    print("Prefixes")
    for dp in dp_collection:
        print(dp)

    # overall_breakdown = {'Embedding': [1.6375999999999995, 1.940399999999999, 37.51233333333331, 48.580400000005966, 0.0452, 19.931, 15.11633333333337, 1.272, 5.6240000000000006], 'Misc': [157.88379999999995, 168.24159999999998, 975.6883333333325, 219.28179999993364, 13.971400000000012, 1179.6393333310275, 1275.7273333315452, 1585.8539999970144, 2038.8949999973183], 'Linear': [4589.285, 5523.2336, 74311.3850000003, 2448.320799999738, 52.21420000000005, 744.9763333334362, 712.7503333333724, 650.2010000000137, 884.7940000000058], 'LayerNorm': [240.6873999999998, 308.93619999999976, 5933.845, 134.73559999999384, 1.0748, 148.74366666664105, 135.07299999998074, 39.62699999999565, 78.63499999999559], 'Attention': [1094.077400000001, 2321.259000000002, 61903.84833333335, 0, 518.8122000000001, 1358.619666666904, 1151.2226666664803, 1066.45099999979, 1508.7879999998086], 'Copy': [5.806999999999997, 6.184599999999991, 16.77133333333333, 61.13080000000135, 0.035, 82.23633333332488, 68.83466666667114, 19.91599999999988, 21.683000000000234], 'Scoring': [1.4188000000000003, 3.7788000000000013, 554.3686666666667, 0, 0, 0, 0, 0, 0], 'Idle': [663.5688000000039, -95.21419999999684, 99642.8929999997, 4444.508600000327, 12.850199999999905, 4023.805333335003, 5026.1766666684125, 6857.360000003351, 10271.482000003074], 'Communication': [0, 0, 0, 0, 0.007000000000000001, 0, 0, 0, 0], '(Preprocessing) Encode Image': [243.603599999995, 244.48139999999475, 0, 0, 0, 0, 0, 0, 0], 'Activation': [0, 0, 0, 8.964200000000227, 0.004, 83.35533333333372, 74.21933333333361, 28.531999999999165, 44.90699999999918], 'Conv1d': [0, 0, 0, 0, 0, 247.71, 641.0923333333346, 0, 1505.7599999999986], 'SinusoidalPositionEncoder': [0, 0, 0, 0, 0, 0.8543333333333326, 2.2880000000000007, 1.5579999999999916, 7.279999999999992], 'HardUpsampling': [0, 0, 0, 0, 0, 0, 2.5396666666665775, 0, 7.722999999999896], 'KV_Cache_Reorder': [0, 0, 0, 0, 0, 8805.019666666996, 8220.238000000203, 7399.892999999837, 7399.967999999807], 'Wav2Vec2FbankFeatureExtractor': [0, 0, 0, 0, 0, 0.002, 0.002, 0, 0], 'Masked_Select': [0, 0, 0, 0, 0, 0, 15.260666666667921, 0, 46.071999999995256]}
    graph_overall(
        overall_breakdown,
        model_name,
        "Workloads",
        "/fsx-atom/yejinlee/analysis_figures/breakdown/overall_breakdown.pdf",
        file_name_passed=True,
    )
    graph_overall_ratio(
        overall_breakdown,
        model_name,
        "Workloads",
        "/fsx-atom/yejinlee/analysis_figures/breakdown/overall_breakdown_normalized.pdf",
        file_name_passed=True,
    )

    del overall_breakdown["Idle"]

    graph_overall(
        overall_breakdown,
        model_name,
        "Workloads",
        "/fsx-atom/yejinlee/analysis_figures/breakdown/overall_breakdown_wo_idle.pdf",
        file_name_passed=True,
    )
    graph_overall_ratio(
        overall_breakdown,
        model_name,
        "Workloads",
        "/fsx-atom/yejinlee/analysis_figures/breakdown/overall_breakdown_wo_idle_normalized.pdf",
        file_name_passed=True,
    )

elif args.figure1_separate:
    # model_name = [
    #     "IT-T\nCM3",
    #     "I-T\nCM3",
    #     "T-I\nCM3",
    #     "T-T\nCodeLlama",
    #     "HSTU",
    #     "S2TT\nSeamless",
    #     "S2ST\nSeamless",
    #     "T2TT\nSeamless",
    #     "T2ST\nSeamless"
    # ]
    work_load = [
        "T-T",
        "IT-T",
        "I-T",
        "T-I",
        "S2TT",
        "S2ST",
        "T2TT",
        "T2ST",
        "",
    ]

    model_name = [
        "Llama",
        "Chameleon",
        "Seamless",
        "HSTU",
    ]

    file_paths = [
        # # Codellama
        # "/fsx-atom/yejinlee/paper_submission_results/bigcode_eval_34B_breakdown/1gpu_1node/MBPP/batch_size_4/",
        # Codellama
        "/fsx-atom/yejinlee/paper_submission_results/bigcode_eval_34B_breakdown/1gpu_1node/HumanEval/batch_size_16/",
        # Chameleon (ImgTxt2Txt)
        "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.16.umca.True.gm.text.ev.False/",
        # # Chameleon (Img2Txt)
        "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.16.umca.True.gm.text.ev.False/",
        # Chameleon (Txt2Img)
        "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.16.en.image_gen.g.True/%j/",
        # Seamless (S2TT)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/S2TT/batch_size_128/",
        # Seamless (S2ST)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/S2ST/batch_size_128/",
        # Seamless (T2TT)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/T2TT/batch_size_384/",
        # Seamless (T2ST)
        "/fsx-atom/yejinlee/paper_submission_results/seamless_breakdown/1gpu_1node/T2ST/batch_size_384/",
        # HSTU
        "/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/profile_results/pytorch/batch_size_32/",
    ]
    desired_prefixes_list = [
        # Codellama
        "MODULE_Embedding_AG*MODULE_LlamaRMSNorm_AG*MODULE_Linear_AG*MODULE_LlamaRotaryEmbedding_AG*MODULE_SiLU_AG*MODULE_TEXT_DECODE_AG*MODULE_Attention_AG*MODULE_SCORING_AG",
        # Chameleon (ImgTxt2Txt)
        "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG",
        # Chameleon (Img2Txt)
        "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG",
        # Chameleon (Txt2Img)
        "MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POST_PROC_IMAGE_DECODE_AG",
        # # Seamless (S2TT)
        "MODULE_TorchSDPA_AG*MODULE_GLU_AG*MODULE_SiLU_AG*MODULE_Wav2Vec2FbankFeatureExtractor_AG*MODULE_KV_Cache_Reorder_AG*MODULE_ReLU_AG*MODULE_Conv1d_AG*MODULE_Linear_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_Dropout_AG*MODULE_StandardEmbedding_AG*MODULE_StandardLayerNorm_AG*MODULE_TiedProjection_AG",
        # Seamless (S2ST)
        "MODULE_StandardLayerNorm_AG*MODULE_SiLU_AG*MODULE_Dropout_AG*MODULE_TiedProjection_AG*MODULE_Conv1d_AG*MODULE_KV_Cache_Reorder_AG*MODULE_Wav2Vec2FbankFeatureExtractor_AG*MODULE_ConvTranspose1d_AG*MODULE_GLU_AG*MODULE_Embedding_AG*MODULE_ReLU_AG*MODULE_Masked_Select_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_HardUpsampling_AG*MODULE_TorchSDPA_AG*MODULE_StandardEmbedding_AG*MODULE_Linear_AG",
        # Seamless (T2TT)
        "MODULE_Dropout_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_TiedProjection_AG*MODULE_ReLU_AG*MODULE_KV_Cache_Reorder_AG*MODULE_Linear_AG*MODULE_TorchSDPA_AG*MODULE_StandardLayerNorm_AG*MODULE_StandardEmbedding_AG",
        # Seamless (T2ST)
        "MODULE_HardUpsampling_AG*MODULE_StandardLayerNorm_AG*MODULE_StandardEmbedding_AG*MODULE_SinusoidalPositionEncoder_AG*MODULE_TiedProjection_AG*MODULE_KV_Cache_Reorder_AG*MODULE_ConvTranspose1d_AG*MODULE_Embedding_AG*MODULE_ReLU_AG*MODULE_Linear_AG*MODULE_Masked_Select_AG*MODULE_Dropout_AG*MODULE_TorchSDPA_AG*MODULE_Conv1d_AG",
        # HSTU
        "MODULE_Embedding_AG*MODULE_Sigmoid_AG*MODULE_LayerNorm_AG*MODULE_Linear_AG*MODULE_Attention_AG",
    ]

    dp_collection = set()
    prefill_overall_breakdown = dict()
    decode_overall_breakdown = dict()
    for k in cmap.keys():
        prefill_overall_breakdown[k] = list()
        decode_overall_breakdown[k] = list()

    assert len(file_paths) == len(desired_prefixes_list)
    for idx, (fp, dp) in enumerate(zip(file_paths, desired_prefixes_list)):
        desired_prefixes = list(set(dp.split("*")))
        sample_breakdown = dict()

        warmup = (
            15
            if len(
                [path for path in glob.glob(DIR_PREFIX + fp + "/profile_sample_15*")]
            )
            > 0
            else (
                0
                if "hstu" in fp
                or fp
                == "/fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
                else 1
            )
        )
        num_sample = (
            3
            if "txt_to_img" in fp
            else len([path for path in glob.glob(DIR_PREFIX + fp + "/profile_sample*")])
        )

        for sample_id in range(warmup, warmup + num_sample):
            if "txt_to_img" in fp:
                _sample_breakdown = dict()
                for i in range(64, 1088, 64):
                    if args.import_pickle:
                        file_path = (
                            DIR_PREFIX
                            + fp
                            + "profile_sample_"
                            + str(sample_id)
                            + "_"
                            + str(i)
                            + "_gpu_0.json.pickle"
                        )
                        if not os.path.isfile(file_path):
                            assert False, file_path
                        f = open(file_path, "rb")
                        profile_result = pickle.load(f)
                        profile_result = process_kernel_breakdown(
                            profile_result["kernel_breakdown"],
                            profile_result["decoding_step_time"],
                            profile_result["gpu_operation_time_per_decoding_step"],
                            profile_result["max_kernel"],
                            profile_result["min_kernel"],
                            plot_graph=False,
                        )
                    else:
                        profile_result = parse_file(
                            fp
                            + "profile_sample_"
                            + str(sample_id)
                            + "_"
                            + str(i)
                            + "_gpu_0.json",
                            plot_graph=False,
                        )
                    gather_result_separate(
                        profile_result, _sample_breakdown, merge_sample=True
                    )
                    # print(">>>> : ", _sample_breakdown)
                if args.import_pickle:
                    file_path = (
                        DIR_PREFIX
                        + fp
                        + (
                            "image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                            if "coco_image" in fp
                            else "image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                        )
                        + "profile_sample_"
                        + str(sample_id)
                        + "_last_gpu_0.json.pickle"
                    )
                    if not os.path.isfile(file_path):
                        assert False, file_path
                    f = open(file_path, "rb")
                    profile_result = pickle.load(f)
                    profile_result = process_kernel_breakdown(
                        profile_result["kernel_breakdown"],
                        profile_result["decoding_step_time"],
                        profile_result["gpu_operation_time_per_decoding_step"],
                        profile_result["max_kernel"],
                        profile_result["min_kernel"],
                        plot_graph=False,
                    )
                else:
                    profile_result = parse_file(
                        fp
                        + (
                            "image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                            if "coco_image" in fp
                            else "image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
                        )
                        + "profile_sample_"
                        + str(sample_id)
                        + "_last_gpu_0.json",
                        plot_graph=False,
                    )
                for k, v in profile_result.items():
                    if k not in _sample_breakdown:
                        profile_result[k] = [0] * 1023 + v
                gather_result_separate(
                    profile_result, _sample_breakdown, merge_sample=True
                )

                gather_result_separate(_sample_breakdown, sample_breakdown)
                # print("FINAL: ", sample_breakdown)
            else:
                if args.import_pickle:
                    file_path = (
                        DIR_PREFIX
                        + fp
                        + "profile_sample_"
                        + str(sample_id)
                        + "_gpu_0.json.pickle"
                    )
                    if not os.path.isfile(file_path):
                        assert False, file_path
                    f = open(file_path, "rb")
                    profile_result = pickle.load(f)
                    profile_result = process_kernel_breakdown(
                        profile_result["kernel_breakdown"],
                        profile_result["decoding_step_time"],
                        profile_result["gpu_operation_time_per_decoding_step"],
                        profile_result["max_kernel"],
                        profile_result["min_kernel"],
                        plot_graph=False,
                    )
                else:
                    profile_result = parse_file(
                        fp + "profile_sample_" + str(sample_id) + "_gpu_0.json",
                        plot_graph=False,
                    )
                # profile_result = {'(Preprocessing) Encode Image': [15.238000000000008, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Misc': [8.823000000000006, 0.43800000000000033, 0.4280000000000003, 0.43200000000000033, 0.4270000000000003, 0.4230000000000003, 0.4230000000000003, 0.4250000000000003, 0.43300000000000033, 0.43400000000000033], 'LayerNorm': [12.997999999999998, 3.6359999999999952, 3.5939999999999954, 3.5929999999999964, 3.5909999999999953, 3.5889999999999964, 3.584999999999997, 3.5929999999999964, 3.586999999999996, 3.5989999999999958], 'Copy': [0.352, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004], 'Scoring': [0.06300000000000001, 0.08300000000000002, 0.08200000000000002, 0.08100000000000002, 0.08300000000000003, 0.08200000000000002, 0.08100000000000002, 0.08500000000000002, 0.08300000000000002, 0.08500000000000002], 'Linear': [315.8960000000001, 40.487999999999936, 40.42199999999994, 40.40599999999994, 40.48999999999995, 40.45299999999991, 40.40899999999992, 40.46499999999993, 40.48299999999995, 40.487999999999914], 'Embedding': [0.092, 0.004, 0.003, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003], 'Attention': [26.062999999999988, 6.31099999999998, 6.221999999999979, 6.220999999999983, 6.248999999999979, 6.246999999999986, 6.252999999999983, 6.2619999999999845, 6.259999999999983, 6.255999999999982], 'Idle': [-9.727000000000032, 97.79900000000009, 92.46100000000007, 103.96900000000008, 90.72800000000007, 83.71700000000011, 107.91000000000008, 97.30100000000007, 98.40400000000008, 106.65400000000011]}
                for idle in profile_result["Idle"]:
                    assert idle >= 0, idle
                gather_result_separate(profile_result, sample_breakdown)

        # sample_breakdown = {'(Preprocessing) Encode Image': [[243.588999999995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [243.62999999999477, 0, 0, 0, 0, 0, 0, 0, 0, 0], [243.59099999999518, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [243.6149999999951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [243.59299999999487, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'Misc': [[151.9389999999998, 0.5760000000000004, 0.5420000000000004, 0.5380000000000004, 0.5420000000000004, 0.5400000000000004, 0.5410000000000004, 0.5390000000000004, 0.5390000000000004, 0.5360000000000004, 0.5410000000000004], [152.19400000000007, 0.5620000000000004, 0.5510000000000004, 0.5420000000000004, 0.5390000000000004, 0.5340000000000004, 0.5360000000000004, 0.5350000000000004, 0.5370000000000004, 0.5450000000000004], [152.20300000000006, 0.5430000000000004, 0.5320000000000004, 0.5400000000000004, 0.5400000000000004, 0.5380000000000004, 0.5450000000000004, 0.5390000000000004, 0.5430000000000004, 0.5390000000000004, 0.5350000000000004, 0.5470000000000004, 0.5420000000000004], [152.166, 0.5560000000000004, 0.5410000000000004, 0.5380000000000004, 0.5440000000000004, 0.5380000000000004, 0.5490000000000004, 0.5450000000000004, 0.5460000000000004, 0.5420000000000004, 0.5400000000000004, 0.5480000000000004], [152.13, 0.5530000000000004, 0.5400000000000004, 0.5440000000000004, 0.5500000000000004, 0.5400000000000004, 0.5410000000000004, 0.5520000000000004, 0.5510000000000004, 0.5450000000000004, 0.5410000000000004, 0.5450000000000004]], 'LayerNorm': [[205.73399999999978, 3.361999999999997, 3.303999999999997, 3.304999999999996, 3.3039999999999954, 3.2999999999999963, 3.3009999999999957, 3.2989999999999964, 3.2999999999999963, 3.300999999999996, 3.3089999999999957], [205.59699999999995, 3.3319999999999963, 3.301999999999995, 3.305999999999996, 3.3029999999999964, 3.3079999999999963, 3.3009999999999957, 3.3019999999999956, 3.2959999999999963, 3.299999999999997], [205.92399999999986, 3.323999999999996, 3.302999999999996, 3.303999999999996, 3.3019999999999956, 3.299999999999996, 3.307999999999996, 3.298999999999996, 3.307999999999996, 3.306999999999996, 3.333999999999995, 3.3009999999999957, 3.291999999999996], [205.5019999999999, 3.3299999999999965, 3.3009999999999957, 3.3039999999999954, 3.2929999999999953, 3.302999999999996, 3.3009999999999953, 3.3009999999999957, 3.301999999999996, 3.3079999999999954, 3.3059999999999965, 3.2999999999999954], [205.4749999999999, 3.3389999999999973, 3.3009999999999953, 3.300999999999996, 3.295999999999996, 3.2929999999999953, 3.295999999999996, 3.306999999999996, 3.3009999999999957, 3.300999999999996, 3.2999999999999954, 3.3039999999999963]], 'Copy': [[5.728, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.008, 0.007], [5.734, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007], [5.7330000000000005, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007], [5.7330000000000005, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007], [5.735000000000001, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007]], 'Scoring': [[0.14500000000000002, 0.11800000000000002, 0.11800000000000005, 0.11700000000000002, 0.11900000000000002, 0.11700000000000002, 0.12200000000000003, 0.12200000000000003, 0.12200000000000003, 0.12000000000000002, 0.12400000000000003], [0.14700000000000002, 0.11700000000000002, 0.11800000000000002, 0.11800000000000002, 0.12000000000000002, 0.12200000000000003, 0.12000000000000002, 0.12000000000000002, 0.11900000000000002, 0.12100000000000002], [0.14600000000000002, 0.11600000000000005, 0.11600000000000002, 0.12000000000000002, 0.11900000000000002, 0.12000000000000002, 0.12100000000000002, 0.11900000000000002, 0.11900000000000002, 0.12200000000000003, 0.12100000000000002, 0.12100000000000002, 0.12300000000000003], [0.14600000000000002, 0.11300000000000004, 0.11800000000000002, 0.12000000000000002, 0.12000000000000002, 0.12100000000000002, 0.12200000000000003, 0.12100000000000002, 0.12000000000000002, 0.135, 0.12000000000000002, 0.12500000000000003], [0.15300000000000002, 0.11300000000000004, 0.11300000000000002, 0.11800000000000002, 0.11900000000000002, 0.11800000000000002, 0.11700000000000002, 0.11900000000000002, 0.12000000000000002, 0.12400000000000003, 0.12700000000000003, 0.12300000000000003]], 'Linear': [[4145.686999999998, 42.18399999999999, 42.03099999999998, 42.136999999999965, 42.09999999999996, 42.17999999999997, 42.09499999999997, 42.163999999999966, 42.24399999999997, 42.12399999999995, 42.28399999999996], [4144.419000000004, 42.075999999999965, 41.43799999999997, 41.43499999999997, 41.89099999999996, 41.48299999999997, 41.74299999999997, 42.147999999999975, 42.57399999999998, 42.48599999999997], [4148.4619999999995, 42.00299999999997, 41.433999999999955, 42.25899999999994, 41.95999999999995, 42.056999999999974, 42.080999999999975, 42.27899999999997, 42.10699999999997, 41.95599999999996, 42.01599999999996, 42.32899999999995, 42.11699999999997], [4140.094, 42.160999999999966, 42.03199999999998, 42.20399999999996, 41.99999999999998, 42.16999999999996, 42.10499999999994, 42.164999999999964, 42.14599999999999, 42.145999999999944, 42.11799999999998, 42.25699999999996], [4135.796000000001, 42.417999999999964, 42.330999999999975, 42.34999999999997, 42.07699999999997, 42.134999999999955, 42.34599999999998, 42.29599999999997, 42.20099999999997, 42.273999999999965, 42.19099999999996, 42.42899999999998]], 'Embedding': [[1.4629999999999999, 0.016, 0.018, 0.016, 0.017, 0.016, 0.016, 0.016, 0.015, 0.016, 0.015], [1.468, 0.017, 0.017, 0.016, 0.017, 0.016, 0.016, 0.016, 0.017, 0.015], [1.4809999999999999, 0.017, 0.018, 0.017, 0.017, 0.015, 0.016, 0.015, 0.015, 0.015, 0.016, 0.015, 0.016], [1.4669999999999999, 0.019, 0.016, 0.017, 0.016, 0.017, 0.016, 0.016, 0.015, 0.016, 0.015, 0.015], [1.4569999999999999, 0.018, 0.015, 0.016, 0.016, 0.015, 0.016, 0.016, 0.016, 0.015, 0.016, 0.015]], 'Attention': [[396.6320000000009, 66.713, 65.36600000000001, 65.41, 65.45900000000005, 65.51700000000008, 65.56800000000005, 65.72800000000004, 65.70300000000003, 65.768, 65.83100000000007], [397.13300000000044, 66.03800000000001, 65.41499999999995, 65.45400000000004, 65.5130000000001, 65.55600000000007, 65.74700000000006, 65.71500000000003, 65.75399999999999, 65.85000000000004], [397.96100000000104, 65.82100000000001, 65.46199999999995, 65.51600000000006, 65.5610000000001, 65.64200000000007, 65.72600000000006, 65.71400000000003, 65.75999999999999, 65.82500000000003, 65.86800000000011, 65.98200000000008, 66.0060000000001], [397.6270000000004, 66.13400000000003, 65.45999999999997, 65.49400000000001, 65.5740000000001, 65.64100000000009, 65.73000000000005, 65.72300000000001, 65.78, 65.82900000000006, 65.89700000000009, 65.96600000000011], [397.55100000000044, 66.20800000000006, 65.46899999999998, 65.48900000000003, 65.5700000000001, 65.61900000000009, 65.72900000000007, 65.733, 65.77599999999994, 65.82000000000002, 65.8880000000001, 65.96600000000011]], 'Idle': [[-241.18599999999242, 73.81700000000004, 65.32200000000002, 70.19600000000005, 62.403999999999996, 73.14599999999994, 72.00999999999999, 75.22099999999999, 69.269, 87.19400000000005, 56.84799999999997], [-241.17500000000018, 94.76100000000002, 168.2050000000001, 160.96699999999998, 92.30199999999994, 144.962, 140.96499999999997, 70.08800000000001, 14.712000000000032, 41.899], [-241.17999999999574, 114.58000000000001, 169.37200000000007, 84.642, 111.04899999999995, 92.86899999999996, 103.24699999999999, 62.062, 113.62800000000004, 97.79400000000003, 107.02099999999993, 68.81699999999995, 91.02999999999993], [-241.18699999999535, 93.44200000000001, 86.95300000000006, 72.40800000000004, 90.83599999999994, 85.11699999999996, 69.44900000000004, 79.21400000000003, 86.15200000000002, 75.892, 83.32399999999993, 73.41699999999992], [-241.18999999999687, 58.14899999999999, 56.14000000000004, 64.584, 74.83599999999993, 77.87199999999996, 71.91199999999996, 76.01400000000004, 70.6440000000001, 78.11700000000002, 80.66899999999993, 68.22199999999992]]}

        if idx < 4:
            prefill_breakdown = dict()
            decode_breakdown = dict()
            print(sample_breakdown)
            for k, v in sample_breakdown.items():
                prefill_breakdown[k] = np.average([vv[0] for vv in v])
                decode_breakdown[k] = np.average([sum(vv[1:]) for vv in v])

            print("prefill: ", prefill_breakdown)
            print("decode: ", decode_breakdown)
            # sample_breakdown = {k: [np.average(v)] for k, v in sample_breakdown.items()}

            # dp_collection = dp_collection.union(set(sample_breakdown.keys()))

            for k, v in prefill_overall_breakdown.items():
                if k in sample_breakdown:
                    prefill_overall_breakdown[k].append(prefill_breakdown[k])
                else:
                    prefill_overall_breakdown[k].append(0)

            for k, v in decode_overall_breakdown.items():
                if k in sample_breakdown:
                    decode_overall_breakdown[k].append(decode_breakdown[k])
                else:
                    decode_overall_breakdown[k].append(0)
        else:
            for k, v in prefill_overall_breakdown.items():
                if k in sample_breakdown:
                    prefill_overall_breakdown[k].append(
                        np.average([sum(vv) for vv in sample_breakdown[k]])
                    )
                    decode_overall_breakdown[k].append(0)
                else:
                    prefill_overall_breakdown[k].append(0)
                    decode_overall_breakdown[k].append(0)

    # prefill_overall_breakdown = {'Embedding': [1.4671999999999998, 1.4791999999999998, 0.030333333333333337, 48.575600000005934, 0.0452, 19.931, 15.11633333333337, 1.272, 5.6240000000000006], 'Misc': [152.1264, 152.202, 2.687666666666662, 165.52639999995858, 13.971400000000012, 1179.6393333310275, 1275.7273333315452, 1585.8539999970144, 2038.8949999973183], 'Linear': [4142.891600000001, 4286.896200000001, 120.80933333333327, 2448.6559999997526, 52.21420000000005, 744.9763333334362, 712.7503333333724, 650.2010000000137, 884.7940000000058], 'LayerNorm': [205.64639999999991, 212.1075999999999, 8.203333333333328, 134.77759999999387, 1.0748, 148.74366666664105, 135.07299999998074, 39.62699999999565, 78.63499999999559], 'Attention': [397.38080000000065, 407.5298000000003, 8.10333333333328, 53.83700000000124, 518.8122000000001, 1358.619666666904, 1151.2226666664803, 1066.45099999979, 1508.7879999998086], 'Copy': [5.732600000000001, 5.981400000000001, 0.09933333333333334, 61.010600000001475, 0.035, 82.23633333332488, 68.83466666667114, 19.91599999999988, 21.683000000000234], 'Scoring': [0.14740000000000003, 0.19540000000000002, 0.6143333333333335, 0, 0, 0, 0, 0, 0], 'Idle': [4905.243000000011, 5066.194199999993, 139.92799999999974, 4077.1308000002864, 12.850199999999905, 4023.805333335003, 5026.1766666684125, 6857.360000003351, 10271.482000003074], 'Communication': [0, 0, 0, 0, 0.007000000000000001, 0, 0, 0, 0], '(Preprocessing) Encode Image': [243.603599999995, 244.48139999999475, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Generate Text': [0, 0, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Decode Image': [0, 0, 0.0, 0, 0, 0, 0, 0, 0], 'Compute': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Activation': [0, 0, 0, 8.966800000000227, 0.004, 83.35533333333372, 74.21933333333361, 28.531999999999165, 44.90699999999918], 'Conv1d': [0, 0, 0, 0, 0, 247.71, 641.0923333333346, 0, 1505.7599999999986], 'SinusoidalPositionEncoder': [0, 0, 0, 0, 0, 0.8543333333333326, 2.2880000000000007, 1.5579999999999916, 7.279999999999992], 'HardUpsampling': [0, 0, 0, 0, 0, 0, 2.5396666666665775, 0, 7.722999999999896], 'KV_Cache_Reorder': [0, 0, 0, 0, 0, 8805.019666666996, 8220.238000000203, 7399.892999999837, 7399.967999999807], 'Wav2Vec2FbankFeatureExtractor': [0, 0, 0, 0, 0, 0.002, 0.002, 0, 0], 'Masked_Select': [0, 0, 0, 0, 0, 0, 15.260666666667921, 0, 46.071999999995256]}
    # decode_overall_breakdown = {'Embedding': [0.17040000000000005, 0.4612000000000003, 42.7903333333341, 0, 0, 0, 0, 0, 0], 'Misc': [5.757400000000004, 16.039600000000014, 1113.6616666666666, 0, 0, 0, 0, 0, 0], 'Linear': [446.39339999999964, 1236.3373999999992, 84979.46833333366, 0, 0, 0, 0, 0, 0], 'LayerNorm': [35.040999999999954, 96.8285999999999, 6774.649999999997, 0, 0, 0, 0, 0, 0], 'Attention': [696.6966000000004, 1913.729200000001, 70531.519, 0, 0, 0, 0, 0, 0], 'Copy': [0.07440000000000001, 0.2032000000000001, 19.132999999999747, 0, 0, 0, 0, 0, 0], 'Scoring': [1.2714, 3.583400000000001, 633.0176666666692, 0, 0, 0, 0, 0, 0], 'Idle': [1184.133199999998, 3263.599199999996, 165569.43599999766, 0, 0, 0, 0, 0, 0], 'Communication': [0, 0, 0, 0, 0, 0, 0, 0, 0], '(Preprocessing) Encode Image': [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Generate Text': [0, 0, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Decode Image': [0, 0, 383.9186666666576, 0, 0, 0, 0, 0, 0], 'Compute': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Activation': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Conv1d': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'SinusoidalPositionEncoder': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'HardUpsampling': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'KV_Cache_Reorder': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Wav2Vec2FbankFeatureExtractor': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Masked_Select': [0, 0, 0, 0, 0, 0, 0, 0, 0]}

    # prefill_overall_breakdown = {'Embedding': [1.4671999999999998, 1.4791999999999998, 0.030333333333333337, 1.5335999999999959, 0.0452, 19.931, 15.11633333333337, 1.272, 5.6240000000000006], 'Misc': [152.1264, 152.202, 2.687666666666662, 31.942199999999996, 13.971400000000012, 1179.6393333310275, 1275.7273333315452, 1585.8539999970144, 2038.8949999973183], 'Linear': [4142.891600000001, 4286.896200000001, 120.80933333333327, 479.8009999999998, 52.21420000000005, 744.9763333334362, 712.7503333333724, 650.2010000000137, 884.7940000000058], 'LayerNorm': [205.64639999999991, 212.1075999999999, 8.203333333333328, 41.39560000000004, 1.0748, 148.74366666664105, 135.07299999998074, 39.62699999999565, 78.63499999999559], 'Attention': [397.38080000000065, 407.5298000000003, 8.10333333333328, 53.83700000000124, 518.8122000000001, 1358.619666666904, 1151.2226666664803, 1066.45099999979, 1508.7879999998086], 'Copy': [5.732600000000001, 5.981400000000001, 0.09933333333333334, 61.010600000001475, 0.035, 82.23633333332488, 68.83466666667114, 19.91599999999988, 21.683000000000234], 'Scoring': [0.14740000000000003, 0.19540000000000002, 0.6143333333333335, 0, 0, 0, 0, 0, 0], 'Idle': [4905.243000000011, 5066.194199999993, 139.92799999999974, 4077.1308000002864, 12.850199999999905, 4023.805333335003, 5026.1766666684125, 6857.360000003351, 10271.482000003074], 'Communication': [0, 0, 0, 0, 0.007000000000000001, 0, 0, 0, 0], '(Preprocessing) Encode Image': [243.603599999995, 244.48139999999475, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Generate Text': [0, 0, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Decode Image': [0, 0, 0.0, 0, 0, 0, 0, 0, 0], 'Compute': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Activation': [0, 0, 0, 8.966800000000227, 0.004, 83.35533333333372, 74.21933333333361, 28.531999999999165, 44.90699999999918], 'Conv1d': [0, 0, 0, 0, 0, 247.71, 641.0923333333346, 0, 1505.7599999999986], 'SinusoidalPositionEncoder': [0, 0, 0, 0, 0, 0.8543333333333326, 2.2880000000000007, 1.5579999999999916, 7.279999999999992], 'HardUpsampling': [0, 0, 0, 0, 0, 0, 2.5396666666665775, 0, 7.722999999999896], 'KV_Cache_Reorder': [0, 0, 0, 0, 0, 8805.019666666996, 8220.238000000203, 7399.892999999837, 7399.967999999807], 'Wav2Vec2FbankFeatureExtractor': [0, 0, 0, 0, 0, 0.002, 0.002, 0, 0], 'Masked_Select': [0, 0, 0, 0, 0, 0, 15.260666666667921, 0, 46.071999999995256]}
    # decode_overall_breakdown = {'Embedding': [0.17040000000000005, 0.4612000000000003, 42.7903333333341, 43.22540000000004, 0, 0, 0, 0, 0], 'Misc': [5.757400000000004, 16.039600000000014, 1113.6616666666666, 181.12159999999872, 0, 0, 0, 0, 0], 'Linear': [446.39339999999964, 1236.3373999999992, 84979.46833333366, 84979.46833333366, 0, 0, 0, 0, 0], 'LayerNorm': [35.040999999999954, 96.8285999999999, 6774.649999999997, 6774.649999999997, 0, 0, 0, 0, 0], 'Attention': [696.6966000000004, 1913.729200000001, 70531.519, 70531.519, 0, 0, 0, 0, 0], 'Copy': [0.07440000000000001, 0.2032000000000001, 19.132999999999747, 19.132999999999747, 0, 0, 0, 0, 0], 'Scoring': [1.2714, 3.583400000000001, 633.0176666666692, 633.0176666666692, 0, 0, 0, 0, 0], 'Idle': [1184.133199999998, 3263.599199999996, 165569.43599999766, 165569.43599999766, 0, 0, 0, 0, 0], 'Communication': [0, 0, 0, 0, 0, 0, 0, 0, 0], '(Preprocessing) Encode Image': [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Generate Text': [0, 0, 0, 0, 0, 0, 0, 0, 0], '(Postprocessing) Decode Image': [0, 0, 383.9186666666576, 0, 0, 0, 0, 0, 0], 'Compute': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Activation': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Conv1d': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'SinusoidalPositionEncoder': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'HardUpsampling': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'KV_Cache_Reorder': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Wav2Vec2FbankFeatureExtractor': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'Masked_Select': [0, 0, 0, 0, 0, 0, 0, 0, 0]}

    if "1gpu_1node" in file_paths[0]:
        if "Communication" in prefill_breakdown:
            del prefill_overall_breakdown["Communication"]
        if "Communication" in decode_overall_breakdown:
            del decode_overall_breakdown["Communication"]

    del_keys = list()
    for k, v in prefill_overall_breakdown.items():
        if (
            sum(v) == 0
            and k in decode_overall_breakdown
            and sum(decode_overall_breakdown[k]) == 0
        ):
            del_keys.append(k)

    for k in del_keys:
        del prefill_overall_breakdown[k]
        del decode_overall_breakdown[k]

    print(prefill_overall_breakdown)
    print(decode_overall_breakdown)

    assert prefill_overall_breakdown.keys() == decode_overall_breakdown.keys()

    key_order = [
        "Attention",
        "Linear",
        "LayerNorm",
        "Conv1d",
        "Embedding",
        "KV_Cache_Reorder",
        "Misc",
        "Idle",
    ]
    print(set(prefill_overall_breakdown.keys()))
    print(set(key_order))
    assert set(prefill_overall_breakdown.keys()) == set(key_order)
    graph_overall_compare_separate(
        prefill_overall_breakdown,
        decode_overall_breakdown,
        key_order,
        work_load,
        model_name,
        "Workloads",
        "./onellm_scripts/analysis_figures/breakdown/separate_overall_breakdown.pdf",
        file_name_passed=True,
    )
    graph_overall_compare_separate_ratio(
        prefill_overall_breakdown,
        decode_overall_breakdown,
        key_order,
        work_load,
        model_name,
        "Workloads",
        "./onellm_scripts/analysis_figures/breakdown/separate_overall_breakdown_ratio.pdf",
        file_name_passed=True,
    )

else:
    if args.json_folder:
        sample_breakdown = dict()
        prefill_overall_breakdown = dict()
        decode_overall_breakdown = dict()
        for k in cmap.keys():
            prefill_overall_breakdown[k] = list()
            decode_overall_breakdown[k] = list()

        print([path for path in glob.glob(args.json_folder + "/profile_sample_*")])
        for file_path in [
            path for path in glob.glob(args.json_folder + "/profile_sample_*")
        ]:
            # if "t2i" in file_path:
            #     sample_id = 500
            #     _sample_breakdown = dict()
            #     # for i in range(64, 1088, 64):
            #     for i in range(64, 129, 64):
            #         profile_result = parse_file(
            #             file_path
            #             + "profile_sample_"
            #             + str(sample_id)
            #             + "_"
            #             + str(i)
            #             + "_gpu_0.json",
            #             file_path,
            #             plot_graph=False,
            #         )
            #         gather_result_separate(
            #             profile_result, _sample_breakdown, merge_sample=True
            #         )

            #     gather_result_separate(_sample_breakdown, sample_breakdown)
            #     print(sample_breakdown)
            #     import pdb

            #     pdb.set_trace()
            #     exit(0)

            # else:
            try:
                profile_result = parse_file(file_path, plot_graph=False)
                # profile_result = {'(Preprocessing) Encode Image': [15.238000000000008, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Misc': [8.823000000000006, 0.43800000000000033, 0.4280000000000003, 0.43200000000000033, 0.4270000000000003, 0.4230000000000003, 0.4230000000000003, 0.4250000000000003, 0.43300000000000033, 0.43400000000000033], 'LayerNorm': [12.997999999999998, 3.6359999999999952, 3.5939999999999954, 3.5929999999999964, 3.5909999999999953, 3.5889999999999964, 3.584999999999997, 3.5929999999999964, 3.586999999999996, 3.5989999999999958], 'Copy': [0.352, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004], 'Scoring': [0.06300000000000001, 0.08300000000000002, 0.08200000000000002, 0.08100000000000002, 0.08300000000000003, 0.08200000000000002, 0.08100000000000002, 0.08500000000000002, 0.08300000000000002, 0.08500000000000002], 'Linear': [315.8960000000001, 40.487999999999936, 40.42199999999994, 40.40599999999994, 40.48999999999995, 40.45299999999991, 40.40899999999992, 40.46499999999993, 40.48299999999995, 40.487999999999914], 'Embedding': [0.092, 0.004, 0.003, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003], 'Attention': [26.062999999999988, 6.31099999999998, 6.221999999999979, 6.220999999999983, 6.248999999999979, 6.246999999999986, 6.252999999999983, 6.2619999999999845, 6.259999999999983, 6.255999999999982], 'Idle': [-9.727000000000032, 97.79900000000009, 92.46100000000007, 103.96900000000008, 90.72800000000007, 83.71700000000011, 107.91000000000008, 97.30100000000007, 98.40400000000008, 106.65400000000011]}
                for idle in profile_result["Idle"]:
                    assert idle >= 0, idle
                gather_result_separate(profile_result, sample_breakdown)
            except:
                print("????")
                pass
        for k, v in sample_breakdown.items():
            print(k, np.average([vv[0] for vv in v]))

        # prefill_breakdown = dict()
        # decode_breakdown = dict()
        # for k, v in sample_breakdown.items():
        #     prefill_breakdown[k] = np.average([vv[0] for vv in v])
        #     decode_breakdown[k] = np.average([sum(vv[1:]) for vv in v])

        # print("Prefill")
        # for k, v in prefill_breakdown.items():
        #     print(k, v)
        # print("Decode")
        # for k, v in decode_breakdown.items():
        #     print(k, v)

    else:
        # if args.json_file:
        folder_name_split = args.json_file.split("/")
        # exp_info = folder_name_split[-1].split("bs")
        # save_folder_path = args.graph_path+"/"+(folder_name_split[-2:] if "retrieval" in args.json_file else folder_name_split[-1])
        save_folder_path = (
            args.graph_path
            + "/"
            + ("wo_efficient_attn/" if "wo_efficient_attn" in args.json_file else "")
            + ("retrieval/" if "retrieval" in args.json_file else "")
            + folder_name_split[-2]
            + "/"
        )
        profile_result = parse_file(args.json_file, save_folder_path=save_folder_path)
        total_time = 0
        for k, v in profile_result.items():
            if k == "Copy":
                continue
            else:
                total_time += sum(v)
        print(total_time)
