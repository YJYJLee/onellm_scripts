# Plot Graph (Overall)
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1
NRETRIEVED_DOCS=$2


# Image to Text
## MSCOCO
TASK="coco.0_shot.flamingo_retrieval_v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}

## Flickr30k
TASK="flickr30k.0_shot.flamingo_retrieval_v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}




# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
NUM_SAMPLE=6
## COCO
TASK="coco_image.0_shot.default_retrieval_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}

## Partiprompts
TASK="partiprompts.0_shot.default_retrieval_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}





# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.flamingo_retrieval_v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}

## TextVQA
TASK="textvqa.0_shot.flamingo_retrieval_v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}

## Vizwiz
TASK="vizwiz.0_shot.flamingo_retrieval_v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}
