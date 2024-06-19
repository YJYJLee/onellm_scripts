python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG --batch-size --multigpu
python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.flickr30k.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG --batch-size --multigpu


python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.okvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG --batch-size --multigpu
python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.textvqa.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG --batch-size --multigpu
python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/img_txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.vizwiz.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_POSTPROC_GENERATE_TEXT_AG --batch-size --multigpu


# python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.coco_image.0_shot.bs.10.c.6.t.1.0.t.0.9.s.1.ncs.32.en.image_gen.g.True --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POSTPROC_GENERATE_TEXT_AG*MODULE_MODEL_PREP_AG*MODULE_POST_PROC_IMAGE_DECODE_AG --batch-size --multigpu
# python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_img/cm3v21_30b_test.mn.cm3v21_30b_test.t.partiprompts.0_shot.cm3v2_template.mbs.1.umca.True.gm.text.ev.False --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG*MODULE_POSTPROC_GENERATE_TEXT_AG*MODULE_MODEL_PREP_AG*MODULE_POST_PROC_IMAGE_DECODE_AG --batch-size --multigpu

python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.arc_easy.0_shot.mbs.1.umca.True.gm.text --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_LOG_SOFTMAX_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_PREPROC_TEXT_ENCODE_AG*MODULE_LOG_SOFTMAX_AG*MODULE_POSTPROC_GENERATE_TEXT_AG --batch-size --multigpu
python onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file /fsx-atom/yejinlee/cm3v2_breakdown_30B_final/1gpu_1node/txt_to_txt/cm3v21_30b_test.mn.cm3v21_30b_test.t.hellaswag.0_shot.mbs.1.umca.True.gm.text --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_LOG_SOFTMAX_AG*MODULE_PREPROC_ENCODE_IMAGES_AG*MODULE_PREPROC_TEXT_ENCODE_AG*MODULE_LOG_SOFTMAX_AG*MODULE_POSTPROC_GENERATE_TEXT_AG --batch-size --multigpu