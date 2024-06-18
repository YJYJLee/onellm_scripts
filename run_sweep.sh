NGPU=$1
NNODE=$2
NTASK=$((NGPU * NNODE))
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok  python stool.py run cm3v21_30b_test onellm_eval/run.py --sweep configs/sweeps/cm3_v2_image.yaml --mem 480 --ncpu 10 --ngpu $NGPU --ntasks $NTASK --nodes $NNODE --partition transformers --account a100-atom --anaconda ~/mambaforge/envs/rag

FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok  python stool.py run cm3v21_30b_test scripts/cm3v2/gen_images.py --sweep configs/sweeps/cm3_v2_image_gen.yaml --mem 480 --ncpu 10 --ngpu 4 --ntasks 4 --nodes 1 --partition transformers --account a100-atom --anaconda ~/mambaforge/envs/rag --image_generation

FSD=/fsx-onellm/data/eval  python stool.py run cm3v21_30b_test onellm_eval/run.py --sweep configs/sweeps/cm3_v2_text.yaml --mem 480 --ncpu 10 --ngpu 2 --ntasks 2 --nodes 1 --partition atom --account a100-atom --anaconda ~/mambaforge/envs/rag
