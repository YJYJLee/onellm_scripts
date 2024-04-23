# Make Graph (Overall Graphs) - w/ and w/o flash attn
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown/"
WO_EFFICIENT_ATTN_BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown/wo_efficient_attn"
# # Image to Text
# ## MSCOCO
# TASK="coco.0_shot.cm3v2_template"
# python ./onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --compare-efficient-attn --compare-dir $WO_EFFICIENT_ATTN_BASEFOLDER
# ## Flickr30k
# TASK="flickr30k.0_shot.cm3v2_template"
# python ./onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --compare-efficient-attn --compare-dir $WO_EFFICIENT_ATTN_BASEFOLDER

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
## COCO
TASK="coco_image.0_shot"
python ./onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --compare-efficient-attn --compare-dir $WO_EFFICIENT_ATTN_BASEFOLDER
## Partiprompts
TASK="partiprompts.0_shot"
python ./onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --compare-efficient-attn --compare-dir $WO_EFFICIENT_ATTN_BASEFOLDER


# # Image+Text to Text
# ## OKVQA
# TASK="okvqa.0_shot.cm3v2_template"
# python ./onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --compare-efficient-attn --compare-dir $WO_EFFICIENT_ATTN_BASEFOLDER
# ## TextVQA
# TASK="textvqa.0_shot.cm3v2_template"
# python ./onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --compare-efficient-attn --compare-dir $WO_EFFICIENT_ATTN_BASEFOLDER
# # Vizwiz
# TASK="vizwiz.0_shot.cm3v2_template"
# python ./onellm_scripts/breakdown_script/parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --compare-efficient-attn --compare-dir $WO_EFFICIENT_ATTN_BASEFOLDER


