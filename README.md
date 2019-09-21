# spiketag

For py37 developer

1. bash anaconda3.sh

2. conda install -c hargup/label/pypi pyopengl-accelerate

## only 2.0.2 is compatabile with current vispy

3. go to vispy: python setup.py develop

4. go to spiketag: python setup.py develop


Make sure CUDA 10.0 is installed (nvcc --version)

5. conda install pytorch torchvision -c pytorch
