# Plot Graph (Overall)
MODELNAME="cm3v21_109m_sft"
BATCHSIZE=$1


# Image to Text
## MSCOCO
TASK="coco.0_shot.cm3v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}

## Flickr30k
TASK="flickr30k.0_shot.cm3v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}




# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
NUM_SAMPLE=6
## COCO
TASK="coco_image.0_shot"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}

## Partiprompts
TASK="partiprompts.0_shot"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}





# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.cm3v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}

## TextVQA
TASK="textvqa.0_shot.cm3v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}

## Vizwiz
TASK="vizwiz.0_shot.cm3v2_template"
python ./onellm_scripts/nsight_compute/memory_bw_extract.py --profile_dir /fsx-atom/yejinlee/cm3v2_memory_bw_profile/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}
