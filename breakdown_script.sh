# Run profile
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown"
# Image to Text
## MSCOCO
TASK="coco.0_shot.cm3v2_template"
mkdir -p $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/stat.txt
## Flickr30k
TASK="flickr30k.0_shot.cm3v2_template"
mkdir -p $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/stat.txt

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
## COCO
TASK="coco_image.0_shot"
mkdir -p $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name text_to_img_coco_image --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate > $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/stat.txt
## Partiprompts
TASK="partiprompts.0_shot"
mkdir -p $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name text_to_img_partiprompts --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate > $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/stat.txt

# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.cm3v2_template"
mkdir -p $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/stat.txt
## TextVQA
TASK="textvqa.0_shot.cm3v2_template"
mkdir -p $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/stat.txt
## Vizwiz
TASK="vizwiz.0_shot.cm3v2_template"
mkdir -p $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/stat.txt






# Make Graph (Individual Graphs)
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown"
# Image to Text
## MSCOCO
TASK="coco.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
## Flickr30k
TASK="flickr30k.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
## COCO
TASK="coco_image.0_shot"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
## Partiprompts
TASK="partiprompts.0_shot"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG

# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
## TextVQA
TASK="textvqa.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG
## Vizwiz
TASK="vizwiz.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK}/profile.json --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG






# Make Graph (Overall Graphs)
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown"
# Image to Text
## MSCOCO
TASK="coco.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
## Flickr30k
TASK="flickr30k.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/i2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
## COCO
TASK="coco_image.0_shot"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
## Partiprompts
TASK="partiprompts.0_shot"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/t2i.$MODELNAME.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size

# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
## TextVQA
TASK="textvqa.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size
## Vizwiz
TASK="vizwiz.0_shot.cm3v2_template"
python ../../parse_json_parallel_barebones_enhanced.py --json-file $BASEFOLDER/it2t.$MODELNAME.bs${BATCHSIZE}.${TASK} --desired-prefixes MODULE_RowParallelLinear_AG*MODULE__InnerAttention_AG*MODULE_LayerNorm_AG*MODULE_ColumnParallelLinear_AG*MODULE_FusedRMSNorm_AG*MODULE_ParallelEmbedding_AG*MODULE_SCORING_AG --batch-size

