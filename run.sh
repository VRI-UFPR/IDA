ORIGINALPATH=$(pwd)
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh

FILE=${ORIGINALPATH}/DUTS-TR
if [ ! -e "${FILE}" ]; then
  wget http://saliencydetection.net/duts/download/DUTS-TR.zip
  unzip DUTS-TR.zip
fi

echo "creating paths_input_mask_output.txt"
echo "creating paths_input_mask.txt"
bash prepare_data.sh

bash create_env.sh

FILE=${ORIGINALPATH}/generative_inpainting/model_logs/release_places2_256
if [ ! -e "${FILE}" ]; then
  echo download DeepFillv2 Places2 weights at https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO?usp=sharing and put it in generative_inpainting/model_logs/
  exit
fi

conda activate genInpaint


export CUDA_VISIBLE_DEVICES=1
cp batch_test2.py generative_inpainting/; cp run_inpaint.sh generative_inpainting/; cd ./generative_inpainting/; bash run_inpaint.sh; cd ..;

CURRENT_PATH_IMAGE="${ORIGINALPATH}/DUTS-TR/DUTS-TR-Image/"
CURRENT_PATH_MASK="${ORIGINALPATH}/DUTS-TR/DUTS-TR-Mask/"
CURRENT_PATH_DST="${ORIGINALPATH}/DUTS-TR/DUTS-TR-Inpainted/"

conda activate ida


cd generate_samples; python computeKnn.py --dataset_file ${ORIGINALPATH}/paths_input_mask.txt --obj_path ${CURRENT_PATH_IMAGE} --obj_mask_path ${CURRENT_PATH_MASK} --bg_path ${CURRENT_PATH_DST}; cd ..;

mkdir -p generate_samples/output/images
mkdir -p generate_samples/output/masks

cd generate_samples; python ida.py --dataset_file ${ORIGINALPATH}/paths_input_mask.txt --obj_path ${CURRENT_PATH_IMAGE} --obj_mask_path ${CURRENT_PATH_MASK} --bg_path ${CURRENT_PATH_DST};cd ..;

cd ${ORIGINALPATH}
