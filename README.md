# spiketag

For py37 developer

1. bash anaconda3.sh

2. conda install -c hargup/label/pypi pyopengl-accelerate

## only 2.0.2 is compatabile with current vispy

3. go to vispy: python setup.py develop

4. go to spiketag: python setup.py develop


Make sure CUDA 10.1 is installed (nvcc --version) and nvidia driver is loaded correctly (nvidia-smi)

5. conda install pytorch torchvision -c pytorch

Others:

6. 
```
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
sudo pip install pyaudio
```
