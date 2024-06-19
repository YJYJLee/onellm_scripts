# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib import colormaps
from math import nan, isnan, ceil
import math

class Radar(object):
    def __init__(self, figure, title, labels=None, rect=None):
        if rect is None:
            # rect = [0.05, 0.05, 0.9, 0.9]
            rect = [0.05, 0, 0.9, 0.9]

        self.n = len(title)
        # self.angles = np.arange(0, 360, 360.0/self.n)
        # self.angles = [18, 90, 162, 234, 306]
        self.angles = [0, 90, 180, 270]

        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14)#, position=(0.1, 0.5))

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        # for ax, angle, label in zip(self.axes, self.angles, labels):
        #     ax.set_rgrids(range(1, 6), angle=angle, labels=label, fontsize=12)
        #     ax.spines['polar'].set_visible(False)
        #     ax.set_ylim(0, 6)
            
        for ax, angle in zip(self.axes, self.angles):
            # ax.set_rgrids(range(1, 6), angle=angle, fontsize=12)
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 6)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
        self.ax.fill(angle, values, alpha=0.1, color=kw['color'])



if __name__ == '__main__':
    # tit = ['Communication\n(GB)\n\n', 'Computation\n(TFLOPs)', 'Memory Capacity\n(GB)', 'Latency\n(ms)', 'GPU Util\n(%)']
    tit = ['Computation\n(TFLOPs)', 'Memory Capacity\n(GB)', 'Latency\n(ms)', 'GPU Util\n(%)']
    
    n_gpu = 1
    num_shot = [0]
    batch_size = [
        {
            'MSCOCO': 1,
            'Vizwiz': 1,
            'Coco_Image': 1,
            'S2ST': 1,
            'HumanEval': 1,
            'MBPP': 1,
        },
        # {
        #     'MSCOCO': 16,
        #     'Vizwiz': 16,
        #     'Coco_Image': 16,
        #     'S2ST': 128,
        #     'HumanEval': 16,
        #     'MBPP': 4,
        # }
    ]

    def get_data(ns, batch_dict, log=False):
        def get_seq_len(file_path, model=None):
            file_path += "/seq_lengths.txt"
            if os.path.isfile(file_path):
                f = open(file_path, "r")
                print("Reading from ", file_path)
                if model == "seamless":
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
                    for k in input_seq_len.keys():
                        input_seq_len[k] = np.average(input_seq_len[k])
                        output_seq_len[k] = np.average(output_seq_len[k])
                        decoding_step[k] = np.average(decoding_step[k])
                    return input_seq_len, output_seq_len, decoding_step
                else:
                    input_seq_len = list()
                    output_seq_len = list()
                    decoding_step = list()
                    for sl in f:
                        slsl = re.sub("\n", "", sl).split("\t")
                        input_seq_len.append(float(slsl[0]))
                        output_seq_len.append(float(slsl[1]))
                        decoding_step.append(float(slsl[2]))
                    return np.average(input_seq_len), np.average(output_seq_len), np.average(decoding_step)
            else:
                print("File doesn't exist: " + file_path)

           

        def get_memory_capacity(file_path):
            mem = list()
            file_path += "/memory_alloc.txt"
            if os.path.isfile(file_path):
                f = open(file_path, "r")
                for sl in f:
                    mem.append(float(sl))
            else:
                print("File doesn't exist: " + file_path)

            return np.average(mem)

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
                            timer_result[h].append(slsl[idx] if not log else np.log(slsl[idx]))
            else:
                print("File doesn't exist: " + file_path)
            return timer_result

        def collect_data(dataset, bs, model=None, n_layer=-1):
            seq_len_working_dir = get_seq_len_folder(dataset)
            input_seq_len, output_seq_len, decoding_steps = get_seq_len(seq_len_working_dir, model=model)

            working_dir = get_folder(dataset, bs)
            print(dataset)
            print("input_seq: ", input_seq_len)
            print("output_seq_len: ", output_seq_len)
            print("decoding_steps: ", decoding_steps)

            if model == "seamless":
                def get_dict_sum(_dict):
                    _sum = 0
                    for k, v in _dict.items():
                        if type(v)==dict:
                            _sum += get_dict_sum(v)
                        else:
                            print(k, " ", v)
                            _sum += v
                    return _sum

                
                result = [
                        # 0, \
                        (get_dict_sum(get_computation("prefill", bs, input_seq_len, model=model, n_layer={"Encoder": 24, "Decoder": 24, "NAR": 6}, task=dataset))+get_dict_sum(get_computation("decode", bs, {'Encoder': 1, 'Decoder': 1, 'Decode': 1, 'NART_Encoder': 1, 'NART_Decoder': 1, 'Vocoder': 1}, model=model, n_layer={"Encoder": 24, "Decoder": 24, "NAR": 6}, task=dataset, decoding_step=decoding_steps["Decoder"])))/1024/1024/1024/1024, \
                        get_memory_capacity(working_dir)/1024/1024/1024, \
                        sum([np.average(v) for v in get_latency(working_dir).values()]), \
                        get_gpu_util(working_dir)]
                
            else:
                result = [
                        # (sum(get_communication("prefill", bs, input_seq_len).values())+sum(get_communication("decode", bs).values())*decoding_steps)/1024/1024/1024, \
                        (sum(get_computation("prefill", bs=bs, seq_len=input_seq_len, n_layer=n_layer).values())+sum(get_computation("decode", bs=bs, n_layer=n_layer).values())*decoding_steps)/1024/1024/1024/1024, \
                        get_memory_capacity(working_dir)/1024/1024/1024, \
                        sum([np.average(v) for v in get_latency(working_dir).values()]), \
                        get_gpu_util(working_dir)]

                result = [0 if isnan(r) else r for r in result]
            return result

        def get_folder(dataset, bs):
            if dataset == "MSCOCO":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Flickr30k":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "TextVQA":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "OKVQA":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Vizwiz":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Hellaswag":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            elif dataset == "Arc_easy":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            elif dataset == "HumanEval":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/HumanEval_codellama/batch_size_"+str(bs)
            elif dataset == "MBPP":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/MBPP_codellama/batch_size_"+str(bs)
            elif dataset == "Coco_Image":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
            elif dataset == "Partiprompts":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
            elif dataset == "S2ST":
                return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/S2ST/batch_size_"+str(bs)+"/"
            else:
                assert False

        def get_seq_len_folder(dataset):
            if dataset == "MSCOCO":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/"
            elif dataset == "Flickr30k":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False"
            elif dataset == "TextVQA":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False"
            elif dataset == "OKVQA":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False"
            elif dataset == "Vizwiz":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False"
            # elif dataset == "Hellaswag":
            #     return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            # elif dataset == "Arc_easy":
            #     return "/fsx-atom/yejinlee/sweep_final/"+str(n_gpu)+"gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            elif dataset == "Coco_Image":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
            elif dataset == "Partiprompts":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1"
            elif dataset == "HumanEval":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/codellama/humaneval"
            elif dataset == "MBPP":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/codellama/mbpp"
            elif dataset == "S2ST":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2ST/"
            elif dataset == "S2TT":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2TT/"
            elif dataset == "T2ST":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2ST/"
            elif dataset == "T2TT":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2TT/"
            else:
                assert False


        def get_communication(phase, bs, seq_len=-1):
            if phase=="prefill":
                return {
                    'Embedding': 2*65536*(8192/n_gpu) * n_gpu,
                    'Attention': bs*seq_len*16384/n_gpu * n_gpu,
                    'FFN': bs*seq_len*16384/n_gpu * n_gpu,
                    'Linear': bs*seq_len*131072/n_gpu * n_gpu,
                }
            elif phase == "decode":
                return {
                    'Embedding': 2*65536*(8192/n_gpu) * n_gpu,
                    'Attention': bs*16384/n_gpu * n_gpu,
                    'FFN': bs*16384/n_gpu * n_gpu,
                    'Linear': bs*131072/n_gpu * n_gpu,
                }
            else:
                assert False

        def get_computation(phase, bs, seq_len=-1, model = None, n_layer=-1, task=None, decoding_step=1):
            if model == "seamless":
                def get_conv1d(IN, OUT, kernel, stride=1, padding=0, dilation=1, groups=1):
                    return 2*(kernel*IN*OUT/groups*bs*(D/stride))

                D=1024
                flops = dict()

                S = seq_len['Encoder']
                num_l = n_layer["Encoder"]
                if task=="S2ST" or task=="S2TT":
                    flops["speech_encoder"] = dict()
                    flops["speech_encoder"]["StandardFeedForwardNetwork1"] = dict()
                    flops["speech_encoder"]["StandardFeedForwardNetwork1"]["input_proj"] = (2*bs*S*4*D*D)*num_l
                    flops["speech_encoder"]["StandardFeedForwardNetwork1"]["output_proj"] = (2*bs*S*4*D*D)*num_l
                    flops["speech_encoder"]["StandardMultiheadAttention"] = (24*bs*S*D*D + 4*bs*S*S*D)*num_l
                    flops["speech_encoder"]["ConformerConvolution"] = dict()
                    IN=1024; OUT=2048
                    flops["speech_encoder"]["ConformerConvolution"]["conv1"] = get_conv1d(IN, OUT, 1, 1)*num_l
                    IN=1024; OUT=1024
                    flops["speech_encoder"]["ConformerConvolution"]["conv2"] = get_conv1d(IN, OUT, 31, 1, groups=1024)*num_l
                    IN=1024; OUT=1024
                    flops["speech_encoder"]["ConformerConvolution"]["conv3"] = get_conv1d(IN, OUT, 1, 1)*num_l
                    flops["speech_encoder"]["StandardFeedForwardNetwork2"] = dict()
                    flops["speech_encoder"]["StandardFeedForwardNetwork2"]["input_proj"] = (2*bs*S*4*D*D)*num_l
                    flops["speech_encoder"]["StandardFeedForwardNetwork2"]["output_proj"] = (2*bs*S*4*D*D)*num_l

                    flops["linear1"] = 8*bs*S*D*D
                    flops["linear2"] = 8*bs*S*D*D

                    flops["UnitYTransformerAdaptorLayer"] = dict()
                    IN=1024; OUT=2048
                    flops["UnitYTransformerAdaptorLayer"]["conv1d"] = get_conv1d(IN, OUT, 8, 8, 4)
                    IN=1024; OUT=2048
                    flops["UnitYTransformerAdaptorLayer"]["conv1d2"] = get_conv1d(IN, OUT, 8, 8, 4)
                    flops["UnitYTransformerAdaptorLayer"]["StandardMultiheadAttention"] = 24*bs*S*D*D + 4*bs*S*S*D
                    flops["UnitYTransformerAdaptorLayer"]["StandardFeedForwardNetwork"] = dict()
                    flops["UnitYTransformerAdaptorLayer"]["StandardFeedForwardNetwork"]["input_proj"] = 2*bs*S*4*D*D
                    flops["UnitYTransformerAdaptorLayer"]["StandardFeedForwardNetwork"]["output_proj"] = 2*bs*S*4*D*D


                if task=="T2TT" or task=="T2ST":
                    flops["t2tt_encoder"] = dict()
                    flops["t2tt_encoder"]["StandardMultiheadAttention"] = (24*bs*S*D*D + 4*bs*S*S*D)*num_l
                    flops["t2tt_encoder"]["StandardFeedForwardNetwork"] = (2*bs*S*8*D*D + 2*bs*S*8*D*D)*num_l


                flops["t2tt_decoder"] = dict()
                S = seq_len['Decoder']
                num_l = n_layer["Decoder"]
                flops["t2tt_decoder"]["StandardMultiheadAttention1"] = (24*bs*S*D*D + 4*bs*S*S*D)*num_l*decoding_step
                flops["t2tt_decoder"]["StandardMultiheadAttention2"] = (24*bs*S*D*D + 4*bs*S*S*D)*num_l*decoding_step
                flops["t2tt_decoder"]["StandardFeedForwardNetwork"] = (2*bs*S*8*D*D + 2*bs*S*8*D*D)*num_l*decoding_step


                if task=="T2ST" or task=="S2ST":
                    flops["nar_t2u"] = dict()
                    S = seq_len['NART_Decoder']
                    num_l = n_layer["NAR"]
                    flops["nar_t2u"]["encoder"] = dict()
                    flops["nar_t2u"]["encoder"]["StandardMultiheadAttention"] = (24*bs*S*D*D + 4*bs*S*S*D)*num_l
                    flops["nar_t2u"]["encoder"]["StandardFeedForwardNetwork"] = (2*bs*S*8*D*D + 2*bs*S*8*D*D)*num_l

                    flops["nar_t2u"]["decoder"] = dict()
                    S = seq_len['NART_Decoder']
                    flops["nar_t2u"]["decoder"]["StandardMultiheadAttention"] = (24*bs*S*D*D + 4*bs*S*S*D)*num_l
                    IN=1024; OUT=1024
                    flops["nar_t2u"]["decoder"]["conv1d"] = get_conv1d(IN, OUT, 7, 1)*num_l
                    IN=1024; OUT=1024
                    flops["nar_t2u"]["decoder"]["conv1d2"] = get_conv1d(IN, OUT, 7, 1)*num_l
                    flops["nar_t2u"]["decoder"]["tiedprojection"] = 2*bs*S*D*10082

                    
                    flops["vocoder"] = dict()
                    S = seq_len['Vocoder']
                    IN=256; OUT=256
                    for i in range(15):
                        flops["vocoder"][i] = dict()

                    flops["vocoder"][0]["convs1"] = get_conv1d(IN, OUT, 3, 1, 1) + \
                                                    get_conv1d(IN, OUT, 3, 1, 3, 3) + \
                                                    get_conv1d(IN, OUT, 3, 1, 5, 5)
                    flops["vocoder"][0]["convs2"] = get_conv1d(IN, OUT, 3, 1, 1)*3
                    flops["vocoder"][1]["convs1"] = get_conv1d(IN, OUT, 7, 1, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 9, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 15, 5)
                    flops["vocoder"][1]["convs2"] = get_conv1d(IN, OUT, 7, 1, 3)*3
                    flops["vocoder"][2]["convs1"] = get_conv1d(IN, OUT, 11, 1, 5) + \
                                                    get_conv1d(IN, OUT, 11, 1, 15, 3) + \
                                                    get_conv1d(IN, OUT, 11, 1, 25, 5)
                    flops["vocoder"][2]["convs2"] = get_conv1d(IN, OUT, 11, 1, 5)*3

                    IN=128; OUT=128
                    flops["vocoder"][3]["convs1"] = get_conv1d(IN, OUT, 3, 1, 1) + \
                                                    get_conv1d(IN, OUT, 3, 1, 3, 3) + \
                                                    get_conv1d(IN, OUT, 3, 1, 5, 5)
                    flops["vocoder"][3]["convs2"] = get_conv1d(IN, OUT, 3, 1, 1)*3
                    flops["vocoder"][4]["convs1"] = get_conv1d(IN, OUT, 7, 1, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 9, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 15, 5)
                    flops["vocoder"][4]["convs2"] = get_conv1d(IN, OUT, 7, 1, 3)*3
                    flops["vocoder"][5]["convs1"] = get_conv1d(IN, OUT, 11, 1, 5) + \
                                                    get_conv1d(IN, OUT, 11, 1, 15, 3) + \
                                                    get_conv1d(IN, OUT, 11, 1, 25, 5)
                    flops["vocoder"][5]["convs2"] = get_conv1d(IN, OUT, 11, 1, 5)*3

                    IN=64; OUT=64
                    flops["vocoder"][6]["convs1"] = get_conv1d(IN, OUT, 3, 1, 1) + \
                                                    get_conv1d(IN, OUT, 3, 1, 3, 3) + \
                                                    get_conv1d(IN, OUT, 3, 1, 5, 5)
                    flops["vocoder"][6]["convs2"] = get_conv1d(IN, OUT, 3, 1, 1)*3
                    flops["vocoder"][7]["convs1"] = get_conv1d(IN, OUT, 7, 1, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 9, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 15, 5)
                    flops["vocoder"][7]["convs2"] = get_conv1d(IN, OUT, 7, 1, 3)*3
                    flops["vocoder"][8]["convs1"] = get_conv1d(IN, OUT, 11, 1, 5) + \
                                                    get_conv1d(IN, OUT, 11, 1, 15, 3) + \
                                                    get_conv1d(IN, OUT, 11, 1, 25, 5)
                    flops["vocoder"][8]["convs2"] = get_conv1d(IN, OUT, 11, 1, 5)*3

                    IN=32; OUT=32
                    flops["vocoder"][9]["convs1"] = get_conv1d(IN, OUT, 3, 1, 1) + \
                                                    get_conv1d(IN, OUT, 3, 1, 3, 3) + \
                                                    get_conv1d(IN, OUT, 3, 1, 5, 5)
                    flops["vocoder"][9]["convs2"] = get_conv1d(IN, OUT, 3, 1, 1)*3
                    flops["vocoder"][10]["convs1"] = get_conv1d(IN, OUT, 7, 1, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 9, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 15, 5)
                    flops["vocoder"][10]["convs2"] = get_conv1d(IN, OUT, 7, 1, 3)*3
                    flops["vocoder"][11]["convs1"] = get_conv1d(IN, OUT, 11, 1, 5) + \
                                                    get_conv1d(IN, OUT, 11, 1, 15, 3) + \
                                                    get_conv1d(IN, OUT, 11, 1, 25, 5)
                    flops["vocoder"][11]["convs2"] = get_conv1d(IN, OUT, 11, 1, 5)*3

                    IN=16; OUT=16
                    flops["vocoder"][12]["convs1"] = get_conv1d(IN, OUT, 3, 1, 1) + \
                                                    get_conv1d(IN, OUT, 3, 1, 3, 3) + \
                                                    get_conv1d(IN, OUT, 3, 1, 5, 5)
                    flops["vocoder"][12]["convs2"] = get_conv1d(IN, OUT, 3, 1, 1)*3
                    flops["vocoder"][13]["convs1"] = get_conv1d(IN, OUT, 7, 1, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 9, 3) + \
                                                    get_conv1d(IN, OUT, 7, 1, 15, 5)
                    flops["vocoder"][13]["convs2"] = get_conv1d(IN, OUT, 7, 1, 3)*3
                    flops["vocoder"][14]["convs1"] = get_conv1d(IN, OUT, 11, 1, 5) + \
                                                    get_conv1d(IN, OUT, 11, 1, 15, 3) + \
                                                    get_conv1d(IN, OUT, 11, 1, 25, 5)
                    flops["vocoder"][14]["convs2"] = get_conv1d(IN, OUT, 11, 1, 5)*3

                    flops["vocoder"]["conv_post"] = get_conv1d(16, 1, 7, 1, 3)

                    flops["vocoder"]["dur_predictor"] = dict()
                    flops["vocoder"]["dur_predictor"]["conv1"] = get_conv1d(1280, 1280, 3, 1)
                    flops["vocoder"]["dur_predictor"]["conv2"] = get_conv1d(1280, 1280, 3, 1)
                    flops["vocoder"]["dur_predictor"]["linear"] = bs*S
                return flops
            
            else:
                if phase=="prefill":
                    return {
                        'Embedding': 0,
                        'Attention': (bs*seq_len*134217728/n_gpu*n_gpu+bs*seq_len*167772168/n_gpu*n_gpu+bs*seq_len*16777216/n_gpu*n_gpu+bs*seq_len*134217728/n_gpu*n_gpu)*n_layer,
                        'FFN': (bs*seq_len*360710144/n_gpu*n_gpu+bs*seq_len*360710144/n_gpu*n_gpu+bs*seq_len*360710144/n_gpu*n_gpu)*n_layer,
                        'Linear': bs*seq_len*524288000/n_gpu*n_gpu if model=="codellama" else bs*seq_len*1073741824/n_gpu*n_gpu ,
                    }
                elif phase == "decode":
                    return {
                        'Embedding': 0,
                        'Attention': (bs*134217728/n_gpu*n_gpu+bs*167772168/n_gpu*n_gpu+bs*16777216/n_gpu*n_gpu+bs*134217728/n_gpu*n_gpu)*n_layer,
                        'FFN': (bs*360710144/n_gpu*n_gpu+bs*360710144/n_gpu*n_gpu+bs*360710144/n_gpu*n_gpu)*n_layer,
                        'Linear': bs*524288000/n_gpu*n_gpu if model=="codellama" else bs*1073741824/n_gpu*n_gpu,
                    }
                else:
                    assert False

        def get_gpu_util(file_path):
            gpu_util = list()
            file_path += "/gpu_util.txt"
            if os.path.isfile(file_path):
                f = open(file_path, "r")
                for sl in f:
                    gpu_util.append(float(sl.split("/")[-1]))
            else:
                print("File doesn't exist: " + file_path)

            return np.average(gpu_util)

        # data = [
        #     ('[I2T] MSCOCO', collect_data('MSCOCO', 16, n_layer=48)),
        #     # ('[I2T] Flickr30k', collect_data('Flickr30k', 16, n_layer=48)),
        #     # ('[IT2T] TextVQA', collect_data('TextVQA', 16, n_layer=48)),
        #     # ('[IT2T] OKVQA', collect_data('OKVQA', 16, n_layer=48)),
        #     ('[IT2T] Vizwiz', collect_data('Vizwiz', 16, n_layer=48)),
        #     ('[T2I] Coco_Image', collect_data('Coco_Image', 16, n_layer=48)),
        #     # ('[T2I] Partiprompts', collect_data('Partiprompts', 16, n_layer=48)),
        #     ('[S2ST] Fleurs', collect_data('S2ST', 128, model="seamless")),
        #     # ('[T2T] Hellaswag', collect_data('Hellaswag')),
        #     # ('[T2T] Arc_easy', collect_data('Arc_easy')),
        #     ('[T2T] HumanEval', collect_data('HumanEval', 16, model="codellama", n_layer=48)),
        # ]

        # data = [
        #     ('[I2T] MSCOCO', collect_data('MSCOCO', 1, n_layer=48)),
        #     ('[IT2T] Vizwiz', collect_data('Vizwiz', 1, n_layer=48)),
        #     ('[T2I] Coco_Image', collect_data('Coco_Image', 1, n_layer=48)),
        #     ('[S2ST] Fleurs', collect_data('S2ST', 1, model="seamless")),
        #     ('[T2T] HumanEval', collect_data('HumanEval', 1, model="codellama", n_layer=48)),
        # ]


        data = [
            ('[I2T] MSCOCO', collect_data('MSCOCO', batch_dict['MSCOCO'], n_layer=48)),
            ('[IT2T] Vizwiz', collect_data('Vizwiz', batch_dict['Vizwiz'], n_layer=48)),
            ('[T2I] Coco_Image', collect_data('Coco_Image', batch_dict['Coco_Image'], n_layer=48)),
            ('[S2ST] Fleurs', collect_data('S2ST', batch_dict['S2ST'], model="seamless")),
            # ('[T2T] HumanEval', collect_data('HumanEval', batch_dict['HumanEval'], model="codellama", n_layer=48)),
            ('[T2T] MBPP', collect_data('MBPP', batch_dict['MBPP'], model="codellama", n_layer=48)),
        ]

        return data

    cmap = colormaps['tab20c'].colors

    colormap = {
        '[I2T] MSCOCO': cmap[0],
        '[I2T] Flickr30k': cmap[1],
        '[IT2T] TextVQA': cmap[4],
        '[IT2T] OKVQA': cmap[5],
        '[IT2T] Vizwiz': cmap[6],
        '[T2I] Coco_Image': cmap[8],
        '[T2I] Partiprompts': cmap[9],
        # '[T2T] Hellaswag': cmap[8],
        # '[T2T] Arc_easy': cmap[9],
        '[S2ST] Fleurs': cmap[12],
        '[T2T] HumanEval': cmap[14],
        '[T2T] MBPP': cmap[14]
    }
    
    gathered_data = dict()
    for ns in num_shot:
        gathered_data[ns] = dict()
        for idx, bs in enumerate(batch_size):
            gathered_data[ns][idx] = get_data(ns, bs, log=True)
        # gathered_data[ns] = get_data(ns)

    # max_data = [-1]*len(tit)
    # for ns in num_shot:
    #     for gd in gathered_data[ns].values():
    #         for idx, d in enumerate([list(i) for i in zip(*[d[1]for d in gd])]):
    #             max_data[idx] = max(max_data[idx], max(d))
    # # for ns in num_shot:
    # #     for idx, d in enumerate([list(i) for i in zip(*[d[1]for d in gathered_data[ns]])]):
    # #         max_data[idx] = max(max_data[idx], max(d))
    # print("MAX: ", max_data)

    for ns in num_shot:
        for idx, bs in enumerate(batch_size):
            data = gathered_data[ns][idx]
            print("num shot: ", ns, bs)
            for d in data:
                print(d)


            min_data = [math.inf]*len(tit)
            max_data = [-1]*len(tit)
            for idx, d in enumerate([list(i) for i in zip(*[d[1]for d in data])]):
                max_data[idx] = max(max_data[idx], max(d))
                min_data[idx] = min(min_data[idx], min(d))
            print("MAX: ", max_data)


            lab = list()
            shift = -1
            for idx, md in enumerate(max_data):
                if idx==2:
                    if min_data[idx] < 0:
                        shift = min_data[idx]*-1
                        md += shift
                    div = 10**(len(str(int(md)))-1)
                    lab.append([int((ceil(md/div)*div)/5*(i+1)) for i in range(5)])

                else:
                    div = 10**(len(str(int(md)))-1)
                    lab.append([int((ceil(md/div)*div)/5*(i+1)) for i in range(5)])


            fig = plt.figure(figsize=(10, 10))#, layout='tight')

            radar = Radar(fig, tit)#, lab)
            
            for d in data:
                # radar.plot([dd/(max_d/5) if dd>0 else (dd+shift)/(max_d/5) for idx, (dd, max_d) in enumerate(zip(d[1], max_data))], '-', lw=2, marker='o', color=colormap[d[0]], alpha=1, label=d[0])
                radar.plot(d[1], '-', lw=2, marker='o', color=colormap[d[0]], alpha=1, label=d[0])


            radar.ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.05))
            # radar.ax.set_title("# Shot "+str(ns) + " / # GPU " + str(n_gpu) + " / Batch size "+str(bs), fontsize=18)
            fig.show()
            # fig.savefig('/fsx-atom/yejinlee/analysis_figures/star_chart/num_shot'+str(ns)+'_bs1.pdf')
            # print("Saving to "+ '/fsx-atom/yejinlee/analysis_figures/star_chart/num_shot'+str(ns)+'_bs1.pdf')
            fig.savefig('/fsx-atom/yejinlee/analysis_figures/star_chart/num_shot'+str(ns)+re.sub(re.compile(r'\s+'), '', str(bs))+'.pdf')
            print("Saving to "+ '/fsx-atom/yejinlee/analysis_figures/star_chart/num_shot'+str(ns)+re.sub(re.compile(r'\s+'), '', str(bs))+'.pdf')


# %%
