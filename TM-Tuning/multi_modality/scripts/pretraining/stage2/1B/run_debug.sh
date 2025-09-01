export MASTER_PORT=12345
export OMP_NUM_THREADS=1


echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=8
NUM_CPU=8


CUDA_VISIBLE_DEVICES=1 python -m pdb tasks/pretrain.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR}