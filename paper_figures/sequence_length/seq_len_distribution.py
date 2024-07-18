# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib import colormaps
from math import nan, isnan, ceil

DIR_PREFIX="/Users/yejinlee/hpca_2025/onellm_scripts/data_for_paper/radar_chart/"

n_gpu = 1

if __name__ == '__main__':

    def get_data(ns, bs):
        # def get_seq_len(file_path):
        #     input_seq_len = list()
        #     output_seq_len = list()
        #     decoding_step = list()
        #     file_path += "/seq_lengths.txt"
        #     if os.path.isfile(file_path):
        #         f = open(file_path, "r")
        #         for sl in f:
        #             slsl = re.sub("\n", "", sl).split("\t")
        #             input_seq_len.append(float(slsl[0]))
        #             output_seq_len.append(float(slsl[1]))
        #             decoding_step.append(float(slsl[2]))
        #     else:
        #         print("File doesn't exist: " + file_path)

        #     return [np.min(input_seq_len), np.max(input_seq_len), np.average(input_seq_len), np.std(input_seq_len), np.min(output_seq_len), np.max(output_seq_len), np.average(output_seq_len), np.average(decoding_step)]

        def get_seq_len(file_path, model=None):
            file_path += "/seq_lengths.txt"
            if os.path.isfile(file_path):
                f = open(file_path, "r")
                print("Reading from ", file_path)
                if model is not None and (model == "seamless" or "hstu" in model):
                    # seq_lens = dict()
                    input_seq_len = dict()
                    output_seq_len = dict()
                    decoding_step = dict()
                    headers = None
                    for idx, sl in enumerate(f):
                        slsl = re.sub("\n", "", sl).split("\t")
                        slsl = [slslsl for slslsl in slsl if slslsl!=""]
                        
                        if idx==0:
                            headers = slsl
                            for slslsl in slsl:
                                # seq_lens[slslsl] = [[],[],[]]
                                input_seq_len[slslsl] = []
                                output_seq_len[slslsl] = []
                                decoding_step[slslsl] = []
                        else:
                            for idx in range(int(len(headers)/3)):
                                input_seq_len[headers[idx*3]].append(float(slsl[idx*3+0]))
                                output_seq_len[headers[idx*3]].append(float(slsl[idx*3+1]))
                                decoding_step[headers[idx*3]].append(float(slsl[idx*3+2]))
                    
                    # for k in input_seq_len.keys():
                    #     input_seq_len[k] = np.average(input_seq_len[k])
                    #     output_seq_len[k] = np.average(output_seq_len[k])
                    #     decoding_step[k] = np.average(decoding_step[k])
                    result_seq_len = dict()
                    for k in input_seq_len.keys():
                        result_seq_len[k] = [np.min(input_seq_len[k]), np.max(input_seq_len[k]), np.average(input_seq_len[k]), np.std(input_seq_len[k]), np.min(output_seq_len[k]), np.max(output_seq_len[k]), np.average(output_seq_len[k]), np.average(decoding_step[k])]
                        # input_seq_len[k] = [np.min(input_seq_len[k]), np.max(input_seq_len[k]), np.average(input_seq_len[k]), np.std(input_seq_len[k])]
                        # output_seq_len[k] = [np.min(output_seq_len[k]), np.max(output_seq_len[k]), np.average(output_seq_len[k])]
                        # decoding_step[k] =  [np.average(decoding_step[k])]
                    print(result_seq_len)
                    return result_seq_len
                else:
                    input_seq_len = list()
                    output_seq_len = list()
                    decoding_step = list()
                    for sl in f:
                        slsl = re.sub("\n", "", sl).split("\t")
                        input_seq_len.append(float(slsl[0]))
                        output_seq_len.append(float(slsl[1]))
                        decoding_step.append(float(slsl[2]))
                    return [np.min(input_seq_len), np.max(input_seq_len), np.average(input_seq_len), np.std(input_seq_len), np.min(output_seq_len), np.max(output_seq_len), np.average(output_seq_len), np.average(decoding_step)]
            else:
                print("File doesn't exist: " + file_path)
                return []



        def collect_data(dataset, model=None):
            working_dir = get_folder(dataset)
            # input_seq_len, output_seq_len, decoding_steps = get_seq_len(working_dir)
            return get_seq_len(working_dir, model=model)

        # def get_folder(dataset):
        #     if dataset == "MSCOCO":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
        #     elif dataset == "Flickr30k":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
        #     elif dataset == "TextVQA":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
        #     elif dataset == "OKVQA":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
        #     elif dataset == "Vizwiz":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
        #     elif dataset == "Hellaswag":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
        #     elif dataset == "Arc_easy":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
        #     elif dataset == "Coco_Image":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
        #     elif dataset == "Partiprompts":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
        #     elif dataset == "S2ST_Fleurs":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2ST/"
        #     elif dataset == "S2TT_Fleurs":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2TT/"
        #     elif dataset == "T2TT_Fleurs":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2TT/"
        #     elif dataset == "T2ST_Fleurs":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2ST/"
        #     elif dataset == "T2T_HumanEval":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/codellama/humaneval/"
        #     elif dataset == "T2T_MBPP":
        #         return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/codellama/mbpp/"
        #     else:
        #         assert False

        def get_folder(dataset, bs=1):
            if dataset == "MSCOCO":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Flickr30k":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "TextVQA":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "OKVQA":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Vizwiz":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Hellaswag":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            elif dataset == "Arc_easy":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            elif dataset == "HumanEval":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/HumanEval_codellama/batch_size_"+str(bs)
            elif dataset == "MBPP":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/MBPP_codellama/batch_size_"+str(bs)
            elif dataset == "Coco_Image":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
            elif dataset == "Partiprompts":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.500.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
            elif dataset == "S2ST":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/S2ST/batch_size_"+str(bs)+"/"
            elif dataset == "S2TT":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/S2TT/batch_size_"+str(bs)+"/"
            elif dataset == "T2ST":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/T2ST/batch_size_"+str(bs)+"/"
            elif dataset == "T2TT":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/radar_chart/"+str(n_gpu)+"gpu_1node/T2TT/batch_size_"+str(bs)+"/"
            elif dataset == "HSTU-Pytorch":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/sweep/pytorch/batch_size_"+str(bs)+"/"
            elif dataset == "HSTU-Triton":
                return DIR_PREFIX+"/fsx-atom/yejinlee/paper_submission_results/hstu_paper_results/sweep/batch_size_"+str(bs)+"/"
            else:
                assert False, dataset

        data = [
            ('[I2T] MSCOCO', collect_data('MSCOCO')),
            ('[I2T] Flickr30k', collect_data('Flickr30k')),
            ('[IT2T] TextVQA', collect_data('TextVQA')),
            ('[IT2T] OKVQA', collect_data('OKVQA')),
            ('[IT2T] Vizwiz', collect_data('Vizwiz')),
            ('[T2I] Coco_Image', collect_data('Coco_Image')),
            ('[T2I] Partiprompts', collect_data('Partiprompts')),
            ('[S2ST] Fleurs', collect_data('S2ST', model="seamless")),
            # ('[S2TT] Fleurs', collect_data('S2TT', model="seamless")),
            # ('[T2TT] Fleurs', collect_data('T2TT', model="seamless")),
            # ('[T2ST] Fleurs', collect_data('T2ST', model="seamless")),
            ('[T2T] HumanEval', collect_data('HumanEval')),
            ('[T2T] MBPP', collect_data('MBPP')),
            # ('[] HSTU', collect_data('HSTU-Pytorch', model="hstu")),
        ]
        # data = [list(i) for i in zip(*data)]
        return data

    seq_lens = get_data(ns=0, bs=1)
    print(f"{'Dataset': <{20}}", f"{'Input(MIN)': <{15}}", f"{'Input(MAX)': <{15}}", f"{'Input(AVG)': <{15}}",  f"{'Input(DEV)': <{15}}", f"{'Output(MIN)': <{15}}", f"{'Output(MAX)': <{15}}", f"{'Output(AVG)': <{15}}",  f"{'DecodingStep': <{15}}")
    for sl in seq_lens:
        if isinstance(sl[1], dict):
            if "HSTU" in sl[0]:
                print('{0: <20}'.format(sl[0]), ''.join(['{0: <16}'.format(ss) for ss in [f"{s:.2f}" for s in sl[1]["l1"]]]))
            elif "Fleurs" in sl[0]:
                print('{0: <20}'.format(sl[0]), ''.join(['{0: <16}'.format(ss) for ss in [f"{s:.2f}" for s in sl[1]["l1"]]]))
        else:
            print('{0: <20}'.format(sl[0]), ''.join(['{0: <16}'.format(ss) for ss in [f"{s:.2f}" for s in sl[1]]]))

# %%
