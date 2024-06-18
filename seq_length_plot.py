# %%
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import os
from matplotlib import colormaps

def get_sequence_lengh(file_path):
    input_seq_len = list()
    output_seq_len = list()

    if os.path.isfile(file_path):
        f = open(file_path, "r")
        for sl in f:
            slsl = re.sub("\n", "", sl).split("\t")
            input_seq_len.append(float(slsl[0]))
            output_seq_len.append(float(slsl[1]))
    else:
        print("File doesn't exist: " + file_path)

    return np.average(input_seq_len), np.average(output_seq_len)

cmap = colormaps['tab20c'].colors

colormap = {
    'img_to_txt' : cmap[0],
    'img_txt_to_txt': cmap[4],
    'txt_to_img': cmap[8],
    'txt_to_txt': cmap[12],
}
    



def get_seq_len(ns):
    input_seq_len = {
        "img_to_txt": [],        # MSCoco, Flickr30k
        "img_txt_to_txt": [],    # OKVQA, TextVQA, Vizwiz
        "txt_to_img": [],        # Coco image, Partiprompts
        "txt_to_txt": []
    }
    output_seq_len = {
        "img_to_txt": [],        # MSCoco, Flickr30k
        "img_txt_to_txt": [],    # OKVQA, TextVQA, Vizwiz
        "txt_to_img": [],        # Coco image, Partiprompts
        "txt_to_txt": []
    }
    mscoco = get_sequence_lengh("/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco."+str(ns)+"_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
    input_seq_len["img_to_txt"].append(("MSCOCO", mscoco[0]))
    output_seq_len["img_to_txt"].append(("MSCOCO", mscoco[1]))
    flickr30k = get_sequence_lengh("/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k."+str(ns)+"_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
    input_seq_len["img_to_txt"].append(("Flickr30k", flickr30k[0]))
    output_seq_len["img_to_txt"].append(("Flickr30k", flickr30k[1]))
    okvqa = get_sequence_lengh("/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa."+str(ns)+"_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
    input_seq_len["img_txt_to_txt"].append(("OKVQA", okvqa[0]))
    output_seq_len["img_txt_to_txt"].append(("Flickr30k", okvqa[1]))
    textvqa = get_sequence_lengh("/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa."+str(ns)+"_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
    input_seq_len["img_txt_to_txt"].append(("TextVQA", textvqa[0]))
    output_seq_len["img_txt_to_txt"].append(("TextVQA", textvqa[1]))
    vizwiz = get_sequence_lengh("/fsx-atom/yejinlee/sweep_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz."+str(ns)+"_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
    input_seq_len["img_txt_to_txt"].append(("Vizwiz", vizwiz[0]))
    output_seq_len["img_txt_to_txt"].append(("Vizwiz", vizwiz[1]))

    coco_image = get_sequence_lengh("/fsx-atom/yejinlee/sweep/txt_to_img/multigpu/8gpu_1node/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.8.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/seq_lengths.txt")
    input_seq_len["txt_to_img"].append(("Coco Image", coco_image[0]))
    output_seq_len["txt_to_img"].append(("Coco Image", coco_image[1]))
    partiprompts = get_sequence_lengh("/fsx-atom/yejinlee/sweep/txt_to_img/multigpu/8gpu_1node/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts."+str(ns)+"_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.32.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.partiprompts.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1/seq_lengths.txt")
    input_seq_len["txt_to_img"].append(("Partiprompts", partiprompts[0]))
    output_seq_len["txt_to_img"].append(("Partiprompts", partiprompts[1]))

    hellaswag = get_sequence_lengh("/fsx-atom/yejinlee/sweep_final/1gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.hellaswag."+str(ns)+"_shot.mbs.1.umca.True.gm.text/seq_lengths.txt")
    input_seq_len["txt_to_txt"].append(("Hellaswag", hellaswag[0]))
    output_seq_len["txt_to_txt"].append(("Hellaswag", hellaswag[1]))
    arc_easy = get_sequence_lengh("/fsx-atom/yejinlee/sweep_final/1gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.arc_easy."+str(ns)+"_shot.mbs.1.umca.True.gm.text/seq_lengths.txt")
    input_seq_len["txt_to_txt"].append(("Arc_easy", arc_easy[0]))
    output_seq_len["txt_to_txt"].append(("Arc_easy", arc_easy[1]))

    return input_seq_len, output_seq_len


num_shot = [0, 2]
fig = plt.figure(figsize=(6, 4), layout='tight')

for id, ns in enumerate(num_shot):
    input_seq_len, output_seq_len = get_seq_len(ns)
    print(input_seq_len)
    assert input_seq_len.keys() == output_seq_len.keys()
    for k, v in input_seq_len.items():
        plt.scatter([x[1] for x in input_seq_len[k]], [x[1] for x in output_seq_len[k]], label=k if id==0 else None, marker="o" if id==0 else "*", color=colormap[k], s=24)
        for idx, (x, y) in enumerate(zip(input_seq_len[k], output_seq_len[k])):
            plt.annotate(x[0], # this is the text
                        (x[1],y[1]), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center

plt.legend()

plt.ylabel('Decoding Steps', fontsize=12)
plt.xlabel('Input Sequence Length', fontsize=12)

plt.grid()
plt.show()
plt.savefig("./seq_len.pdf")


# %%
