python onellm_scripts/multinode_time_extract.py --task img_to_txt --template cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False
python onellm_scripts/multinode_time_extract.py --task img_to_txt --template cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False

python onellm_scripts/multinode_time_extract.py --task img_txt_to_txt --template cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False
python onellm_scripts/multinode_time_extract.py --task img_txt_to_txt --template cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False
python onellm_scripts/multinode_time_extract.py --task img_txt_to_txt --template cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False

# python onellm_scripts/multinode_time_extract.py --task txt_to_img --template cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1
# python onellm_scripts/multinode_time_extract.py --task txt_to_img --template cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.1.en.image_gen.g.True/%j/image_gen/mn.cm3v21_30b_test.t.coco_image.0_shot.usecfg.True.cfg.6.temp.1.0.topp.0.9.seed.1
