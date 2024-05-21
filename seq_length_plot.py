# %%
import matplotlib.pyplot as plt
import numpy as np
import regex as re

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

input_seq_len = {
    "img_to_txt": [],        # MSCoco, Flickr30k
    "img_txt_to_txt": [],    # OKVQA, TextVQA, Vizwiz
    # "txt_to_img": []        # Coco image, Partiprompts
}
output_seq_len = {
    "img_to_txt": [],        # MSCoco, Flickr30k
    "img_txt_to_txt": [],    # OKVQA, TextVQA, Vizwiz
    # "txt_to_img": []        # Coco image, Partiprompts
}

mscoco = get_sequence_lengh("/fsx-checkpoints/yejinlee/sweep/img_to_txt/multigpu/1gpu_1node/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
input_seq_len["img_to_txt"].append(mscoco[0])
output_seq_len["img_to_txt"].append(mscoco[1])
flickr30k = get_sequence_lengh("/fsx-checkpoints/yejinlee/sweep/img_to_txt/multigpu/1gpu_1node/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
input_seq_len["img_to_txt"].append(flickr30k[0])
output_seq_len["img_to_txt"].append(flickr30k[1])
okvqa = get_sequence_lengh("/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/multigpu/1gpu_1node/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
input_seq_len["img_txt_to_txt"].append(okvqa[0])
output_seq_len["img_txt_to_txt"].append(okvqa[1])
textvqa = get_sequence_lengh("/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/multigpu/1gpu_1node/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
input_seq_len["img_txt_to_txt"].append(textvqa[0])
output_seq_len["img_txt_to_txt"].append(textvqa[1])
vizwiz = get_sequence_lengh("/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/multigpu/1gpu_1node/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False/seq_lengths.txt")
input_seq_len["img_txt_to_txt"].append(vizwiz[0])
output_seq_len["img_txt_to_txt"].append(vizwiz[1])
# coco_image = get_sequence_lengh("/fsx-checkpoints/yejinlee/sweep/txt_to_img/multigpu/1gpu_1node/")
# input_seq_len["txt_to_img"].append(coco_image[0])
# output_seq_len["txt_to_img"].append(coco_image[1])
# partiprompts = get_sequence_lengh("/fsx-checkpoints/yejinlee/sweep/txt_to_img/multigpu/1gpu_1node/")
# input_seq_len["txt_to_img"].append(partiprompts[0])
# output_seq_len["txt_to_img"].append(partiprompts[1])



assert input_seq_len.keys() == output_seq_len.keys()
for k, v in input_seq_len.items():
    plt.scatter(input_seq_len[k], output_seq_len[k], label=k)

plt.legend()
plt.show()
plt.savefig("./test.png")


# %%
