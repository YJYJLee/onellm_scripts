# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib import colormaps
from math import nan, isnan, ceil


if __name__ == '__main__':

    def get_data(ns, bs):
        def get_seq_len(file_path):
            input_seq_len = list()
            output_seq_len = list()
            decoding_step = list()
            file_path += "/seq_lengths.txt"
            if os.path.isfile(file_path):
                f = open(file_path, "r")
                for sl in f:
                    slsl = re.sub("\n", "", sl).split("\t")
                    input_seq_len.append(float(slsl[0]))
                    output_seq_len.append(float(slsl[1]))
                    decoding_step.append(float(slsl[2]))
            else:
                print("File doesn't exist: " + file_path)

            return [np.min(input_seq_len), np.max(input_seq_len), np.average(input_seq_len), np.min(output_seq_len), np.max(output_seq_len), np.average(output_seq_len), np.average(decoding_step)]

        def collect_data(dataset):
            working_dir = get_folder(dataset)
            # input_seq_len, output_seq_len, decoding_steps = get_seq_len(working_dir)
            return get_seq_len(working_dir)

        def get_folder(dataset):
            if dataset == "MSCOCO":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Flickr30k":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "TextVQA":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "OKVQA":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Vizwiz":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs."+str(bs)+".umca.True.gm.text.ev.False/"
            elif dataset == "Hellaswag":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            elif dataset == "Arc_easy":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs."+str(bs)+".umca.True.gm.text/"
            elif dataset == "Coco_Image":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
            elif dataset == "Partiprompts":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/chameleon/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs."+str(bs)+".en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/"
            elif dataset == "S2ST_Fleurs":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2ST/"
            elif dataset == "S2TT_Fleurs":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2TT/"
            elif dataset == "T2TT_Fleurs":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2TT/"
            elif dataset == "T2ST_Fleurs":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2ST/"
            elif dataset == "T2T_HumanEval":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/codellama/humaneval/"
            elif dataset == "T2T_MBPP":
                return "/fsx-atom/yejinlee/paper_submission_results/sequence_lengths/codellama/mbpp/"
            else:
                assert False

        data = [
            ('[I2T] MSCOCO', collect_data('MSCOCO')),
            ('[I2T] Flickr30k', collect_data('Flickr30k')),
            ('[IT2T] TextVQA', collect_data('TextVQA')),
            ('[IT2T] OKVQA', collect_data('OKVQA')),
            ('[IT2T] Vizwiz', collect_data('Vizwiz')),
            ('[T2I] Coco_Image', collect_data('Coco_Image')),
            ('[T2I] Partiprompts', collect_data('Partiprompts')),
            ('[S2ST] Fleurs', collect_data('S2ST_Fleurs')),
            ('[S2TT] Fleurs', collect_data('S2TT_Fleurs')),
            ('[T2TT] Fleurs', collect_data('T2TT_Fleurs')),
            ('[T2ST] Fleurs', collect_data('T2ST_Fleurs')),
            ('[T2T] HumanEval', collect_data('T2T_HumanEval')),
            ('[T2T] MBPP', collect_data('T2T_MBPP')),
            # ('[T2T] Hellaswag', collect_data('Hellaswag')),
            # ('[T2T] Arc_easy', collect_data('Arc_easy')),
        ]
        # data = [list(i) for i in zip(*data)]
        return data

    seq_lens = get_data(ns=0, bs=1)
    print(f"{'Dataset': <{20}}", f"{'Input(MIN)': <{15}}", f"{'Input(MAX)': <{15}}", f"{'Input(AVG)': <{15}}", f"{'Output(MIN)': <{15}}", f"{'Output(MAX)': <{15}}", f"{'Output(AVG)': <{15}}",  f"{'DecodingStep': <{15}}")
    for sl in seq_lens:
        print('{0: <20}'.format(sl[0]), ''.join(['{0: <16}'.format(ss) for ss in [f"{s:.2f}" for s in sl[1]]]))
