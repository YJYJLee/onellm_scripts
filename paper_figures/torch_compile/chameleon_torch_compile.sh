# Img2Txt, ImgTxt2Txt
TORCH_COMPILE=True FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python stool.py run compile_test onellm_eval/run.py --sweep configs/sweeps/cm3_v2_image.yaml --mem 480 --ncpu 10 --ngpu 1 --ntasks 1 --nodes 1 --account atom --anaconda /fsx-atom/yejinlee/condaenv/onellm3
# Txt2Img
TORCH_COMPILE=True FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python stool.py run compile_test scripts/cm3v2/gen_images.py --sweep configs/sweeps/cm3_v2_image_gen.yaml --mem 480 --ncpu 10 --ngpu 1 --ntasks 1 --nodes 1 --account atom --anaconda /fsx-checkpoints/yejinlee/condaenv/onellm3 --image_generation

## Baseline
# Img2Txt, ImgTxt2Txt
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python stool.py run compile_test onellm_eval/run.py --sweep configs/sweeps/cm3_v2_image.yaml --mem 480 --ncpu 10 --ngpu 1 --ntasks 1 --nodes 1 --account atom --anaconda /fsx-atom/yejinlee/condaenv/onellm3
# Txt2Img
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python stool.py run compile_test scripts/cm3v2/gen_images.py --sweep configs/sweeps/cm3_v2_image_gen.yaml --mem 480 --ncpu 10 --ngpu 1 --ntasks 1 --nodes 1 --account atom --anaconda /fsx-checkpoints/yejinlee/condaenv/onellm3 --image_generation
