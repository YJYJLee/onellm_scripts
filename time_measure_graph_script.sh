# Plot Graph (Overall)
MODELNAME="cm3v21_109m_sft"

# Image to Text
## MSCOCO
python ./onellm_scripts/time_extract.py --task img_to_txt --template coco.0_shot.flamingo_template --retrieval-template coco.0_shot.flamingo_retrieval_v2_template --model $MODELNAME

## Flickr30k
python ./onellm_scripts/time_extract.py --task img_to_txt --template flickr30k.0_shot.flamingo_template --retrieval-template flickr30k.0_shot.flamingo_retrieval_v2_template --model $MODELNAME


# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
NUM_SAMPLE=6
## COCO
python ./onellm_scripts/time_extract.py --task txt_to_img --template coco_image.0_shot --retrieval-template coco_image.0_shot.default_retrieval_template --model $MODELNAME

## Partiprompts
python ./onellm_scripts/time_extract.py --task txt_to_img --template partiprompts.0_shot --retrieval-template partiprompts.0_shot.default_retrieval_template --model $MODELNAME


# Image+Text to Text
## OKVQA
python ./onellm_scripts/time_extract.py --task img_txt_to_txt --template okvqa.0_shot.flamingo_template --retrieval-template okvqa.0_shot.flamingo_retrieval_v2_template --model $MODELNAME

## TextVQA
python ./onellm_scripts/time_extract.py --task img_txt_to_txt --template textvqa.0_shot.flamingo_template --retrieval-template textvqa.0_shot.flamingo_retrieval_v2_template --model $MODELNAME

## Vizwiz
python ./onellm_scripts/time_extract.py --task img_txt_to_txt --template vizwiz.0_shot.flamingo_template --retrieval-template vizwiz.0_shot.flamingo_retrieval_v2_template --model $MODELNAME

