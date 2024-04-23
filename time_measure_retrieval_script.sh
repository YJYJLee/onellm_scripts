# Run profile
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1
NRETRIEVED_DOCS=$2

# Image to Text
## MSCOCO
TASK="coco.0_shot.flamingo_retrieval_v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_to_txt/${TASK}.${MODELNAME}/retrieval
EXPDIR=$BASEFOLDER/bs${BATCHSIZE}.n_retrieved_doc${NRETRIEVED_DOCS}
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --n-retrieved-docs $NRETRIEVED_DOCS --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi

## Flickr30k
TASK="flickr30k.0_shot.flamingo_retrieval_v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_to_txt/${TASK}.${MODELNAME}/retrieval
EXPDIR=$BASEFOLDER/bs${BATCHSIZE}.n_retrieved_doc${NRETRIEVED_DOCS}
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --n-retrieved-docs $NRETRIEVED_DOCS --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" >  $EXPDIR/time.txt
fi

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
NUM_SAMPLE=6
## COCO
TASK="coco_image.0_shot.default_retrieval_template"
EXP_NAME=text_to_img_coco_image
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/txt_to_img/${TASK}.${MODELNAME}/retrieval
EXPDIR=${BASEFOLDER}/bs${BATCHSIZE}.n_retrieved_doc${NRETRIEVED_DOCS}
CHECKDIR=${EXPDIR}/$EXP_NAME/mn.${MODELNAME}.t.${TASK}.usecfg.True.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}
if [ -d "$CHECKDIR/images" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME --model-name $MODELNAME --batch-size $NUM_SAMPLE --tasks $TASK --exp_dir $EXPDIR --n-retrieved-docs $NRETRIEVED_DOCS --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate > $EXPDIR/time.txt
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
TASK="partiprompts.0_shot.default_retrieval_template"
EXP_NAME=text_to_img_partiprompts
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/txt_to_img/${TASK}.${MODELNAME}/retrieval
EXPDIR=${BASEFOLDER}/bs${BATCHSIZE}.n_retrieved_doc${NRETRIEVED_DOCS}
CHECKDIR=${EXPDIR}/$EXP_NAME/mn.${MODELNAME}.t.${TASK}.usecfg.True.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}
if [ -d "$CHECKDIR/images" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME --model-name $MODELNAME --batch-size $NUM_SAMPLE --tasks $TASK --exp_dir $EXPDIR --n-retrieved-docs $NRETRIEVED_DOCS --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate > $EXPDIR/time.txt
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
TASK="okvqa.0_shot.flamingo_retrieval_v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/${TASK}.${MODELNAME}/retrieval
EXPDIR=$BASEFOLDER/bs${BATCHSIZE}.n_retrieved_doc${NRETRIEVED_DOCS}
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --n-retrieved-docs $NRETRIEVED_DOCS --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi

## TextVQA
TASK="textvqa.0_shot.flamingo_retrieval_v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/${TASK}.${MODELNAME}/retrieval
EXPDIR=$BASEFOLDER/bs${BATCHSIZE}.n_retrieved_doc${NRETRIEVED_DOCS}
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --n-retrieved-docs $NRETRIEVED_DOCS --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi

## Vizwiz
TASK="vizwiz.0_shot.flamingo_retrieval_v2_template"
BASEFOLDER=/fsx-checkpoints/yejinlee/sweep/img_txt_to_txt/${TASK}.${MODELNAME}/retrieval
EXPDIR=$BASEFOLDER/bs${BATCHSIZE}.n_retrieved_doc${NRETRIEVED_DOCS}
if [ -d "$EXPDIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXPDIR}"
else
    mkdir -p $EXPDIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $EXPDIR --n-retrieved-docs $NRETRIEVED_DOCS --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXPDIR/time.txt
fi




