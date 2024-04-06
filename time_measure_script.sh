# Run profile
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1
# Image to Text
## MSCOCO
TASK="coco.0_shot.cm3v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_to_txt/${TASK}.${MODELNAME}/bs${BATCHSIZE}
EXPDIR=$BASEFOLDER/original
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi

## Flickr30k
TASK="flickr30k.0_shot.cm3v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_to_txt/${TASK}.${MODELNAME}/bs${BATCHSIZE}
EXPDIR=$BASEFOLDER/original
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" >  $EXPDIR/time.txt
fi

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
NUM_SAMPLE=6
## COCO
TASK="coco_image.0_shot"
EXP_NAME=text_to_img_coco_image
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/txt_to_img/${TASK}.${MODELNAME}/bs${BATCHSIZE}
EXPDIR=${BASEFOLDER}/original
CHECKDIR=${EXPDIR}/$EXP_NAME/mn.${MODELNAME}.t.${TASK}.usecfg.True.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}
if [ -d "$CHECKDIR/images" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME --model-name $MODELNAME --batch-size $NUM_SAMPLE --tasks $TASK --exp_dir $EXPDIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate > $EXPDIR/time.txt
fi
if [[ (( -d "$CHECKDIR/raw_results" )) && (( -d "$CHECKDIR/results" )) ]] ; then
    echo "`date` Skipping Scoring... ${EXPDIR}"
else
    mv $CHECKDIR/images/chunk_1.jsonl $CHECKDIR/images/chunk_0.jsonl
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME \
        --model-name $MODELNAME --batch-size $NUM_SAMPLE --tasks $TASK \
        --exp_dir $EXPDIR --cfg $CFG --temp  $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --score > $EXPDIR/score.txt
fi

## Partiprompts
TASK="partiprompts.0_shot"
EXP_NAME=text_to_img_partiprompts
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/txt_to_img/${TASK}.${MODELNAME}/bs${BATCHSIZE}
EXPDIR=${BASEFOLDER}/original
CHECKDIR=${EXPDIR}/$EXP_NAME/mn.${MODELNAME}.t.${TASK}.usecfg.True.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}
if [ -d "$CHECKDIR/images" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME --model-name $MODELNAME --batch-size $NUM_SAMPLE --tasks $TASK --exp_dir $EXPDIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate > $EXPDIR/time.txt
fi
if [[ (( -d "$CHECKDIR/raw_results" )) && (( -d "$CHECKDIR/results" )) ]] ; then
    echo "`date` Skipping Scoring... ${EXPDIR}"
else
    mv $CHECKDIR/images/chunk_1.jsonl $CHECKDIR/images/chunk_0.jsonl
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME \
        --model-name $MODELNAME --batch-size $NUM_SAMPLE --tasks $TASK \
        --exp_dir $EXPDIR --cfg $CFG --temp  $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --score > $EXPDIR/score.txt
fi

# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.cm3v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/${TASK}.${MODELNAME}/bs${BATCHSIZE}
EXPDIR=$BASEFOLDER/original
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi
## TextVQA
TASK="textvqa.0_shot.cm3v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/${TASK}.${MODELNAME}/bs${BATCHSIZE}
EXPDIR=$BASEFOLDER/original
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi
## Vizwiz
TASK="vizwiz.0_shot.cm3v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/${TASK}.${MODELNAME}/bs${BATCHSIZE}
EXPDIR=$BASEFOLDER/original
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi





# # Make Graph (Individual Graphs)
# MODELNAME="cm3v21_109m_sft"
# BATCHSIZE=$1
# BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown"
# # Image to Text
# ## MSCOCO
# TASK="coco.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
# ## Flickr30k
# TASK="flickr30k.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG

# # Text to Image
# CFG=6
# TEMP=1.0
# TOPP=0.9
# SEED=1
# ## COCO
# TASK="coco_image.0_shot"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
# ## Partiprompts
# TASK="partiprompts.0_shot"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG

# # Image+Text to Text
# ## OKVQA
# TASK="okvqa.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
# ## TextVQA
# TASK="textvqa.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
# ## Vizwiz
# TASK="vizwiz.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG






# # Make Graph (Overall Graphs)
# MODELNAME="cm3v21_109m_sft"
# BATCHSIZE=$1
# BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown"
# # Image to Text
# ## MSCOCO
# TASK="coco.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
# ## Flickr30k
# TASK="flickr30k.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size

# # Text to Image
# CFG=6
# TEMP=1.0
# TOPP=0.9
# SEED=1
# ## COCO
# TASK="coco_image.0_shot"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
# ## Partiprompts
# TASK="partiprompts.0_shot"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size

# # Image+Text to Text
# ## OKVQA
# TASK="okvqa.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
# ## TextVQA
# TASK="textvqa.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
# ## Vizwiz
# TASK="vizwiz.0_shot.cm3v2_template"
# python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size

