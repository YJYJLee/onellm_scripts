# Run profile
MODELNAME="cm3v21_109m_sft"
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_breakdown/retrieval"
BATCHSIZE=$1
NRETRIEVED_DOCS=$2
# Image to Text
# MSCOCO
TASK="coco.0_shot.flamingo_retrieval_v2_template"
mkdir -p $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS > $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/stat.txt
## Flickr30k
TASK="flickr30k.0_shot.flamingo_retrieval_v2_template"
mkdir -p $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS > $BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/stat.txt

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
## COCO
TASK="coco_image.0_shot.default_retrieval_template"
mkdir -p $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name text_to_img_coco_image --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS} --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --n-retrieved-docs $NRETRIEVED_DOCS > $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}/stat.txt
## Partiprompts
TASK="partiprompts.0_shot.default_retrieval_template"
mkdir -p $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name text_to_img_partiprompts --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS} --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --n-retrieved-docs $NRETRIEVED_DOCS > $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}/stat.txt

# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.flamingo_retrieval_v2_template"
mkdir -p $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS > $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/stat.txt
## TextVQA
TASK="textvqa.0_shot.flamingo_retrieval_v2_template"
mkdir -p $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS > $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/stat.txt
## Vizwiz
TASK="vizwiz.0_shot.flamingo_retrieval_v2_template"
mkdir -p $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS} --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS > $BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/stat.txt