CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh

FILE=${CONDA_BASE}/envs/genInpaint
if [ ! -e "${FILE}" ]; then
  conda create --name genInpaint python=3.6.12
  conda activate genInpaint
  conda install tensorflow-gpu==1.7.0
  pip install git+https://github.com/JiahuiYu/neuralgym
  pip install opencv-python
  pip install pillow
fi

FILE=${CONDA_BASE}/envs/ida
if [ ! -e "${FILE}" ]; then
  conda create --name ida python=3.6.12
  conda activate ida
  pip install -r requirements.txt
fi
