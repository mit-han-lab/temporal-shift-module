# TSM Online Hand Gesture Recognition Demo

```
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

![tsm-demo](https://file.lzhu.me/projects/tsm/external/tsm-demo2.gif)

See the [[full video]](https://hanlab.mit.edu/projects/tsm/#live_demo) of our demo on NVIDIA Jetson Nano.

**[NEW!]** We have updated the environment set up by using `onnx-simplifier`, which makes the deployment easy. Thanks for the advice from @poincarelee!

## Overview

We show how to deploy an online hand gesture recognition system on **NVIDIA Jetson Nano**. The model is based on MobileNetV2 backbone with **Temporal Shift Module (TSM)** to model the temporal relationship. It is compiled with **TVM** [1] for acceleration. 

The model can achieve **real-time** recognition. Without considering the data IO time, it can achieve **>70 FPS** on Nano GPU.

[1] Tianqi Chen *et al.*, *TVM: An automated end-to-end optimizing compiler for deep learning*, in OSDI 2018

## Model

We used an online version of Temporal Shift Module in this demo. The model design is shown below:
<p align="center">
	<img src="https://hanlab.mit.edu/projects/tsm/external/tsm-online-model.png" width="550">
</p>

After compiled with TVM, our model can efficient run on low-power devices.

<p align="center">
	<img src="https://hanlab.mit.edu/projects/tsm/external/tsm-low-power.png" width="550">
</p>

## Step-by-step Tutorial

We show how to set up the environment on Jetson Nano, compile the PyTorch model with TVM, and perform the online demo from camera streaming.

1. Get an [NVIDIA Jeston Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) board (it is only $99!).
2. Get a micro SD card and burn the **Nano system image** into it following [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit). Insert the card and boot the Nano. **Note**: you may want to get a power adaptor for a stable power supply.
3. Check if OpenCv 4.X is installed (it is now included in SD card image from r32.3.1)
```
 $ Python3
 >> Import cv2
 >> cv2.__version__
```
 It should show 4.X.
 If not, build **OpenCV** 4.0.0 using [this script](https://github.com/AastaNV/JEP/blob/master/script/install_opencv4.0.0_Nano.sh), so that we can enable camera access (It may take a while due to the weak CPU). You also need add cv2 package to path import search path.

```
export PYTHONPATH=/usr/local/python
```

4. Follow [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/) to install **PyTorch** and **torchvision**.
5. Build **TVM** with following commands

```
sudo apt install llvm # install llvm which is required by tvm
git clone -b v0.6 https://github.com/apache/incubator-tvm.git
cd incubator-tvm
git submodule update --init
mkdir build
cp cmake/config.cmake build/
cd build
#[
#edit config.cmake to change
# 32 line: USE_CUDA OFF -> USE_CUDA ON
#104 line: USE_LLVM OFF -> USE_LLVM ON
#]
cmake ..
make -j4
cd ..
cd python; sudo python3 setup.py install; cd ..
cd topi/python; sudo python3 setup.py install; cd ../..
```

6. Install **ONNX**

```
# install onnx
sudo apt-get install protobuf-compiler libprotoc-dev
pip3 install onnx
```

7. Install **onnx-simplifier**

```
git clone https://github.com/daquexian/onnx-simplifier
cd onnx-simplifier
# remove requirement 'onnxruntime >= 1.2.0' in setup.py, as it is not actually used
pip install .
cd ..
```

8. export cuda toolkit binary to path

```
export PATH=$PATH:/usr/local/cuda/bin
```

8. **Finally, run the demo**. The first run will compile the PyTorch TSM model into TVM binary first and then run it. Later run will directly execute the compiled TVM model.

```
python3 main.py
```

Press `Q` or `Esc` to quit. Press `F` to enter/exit full-screen.

## Supported Gestures

- No gesture
- Stop Sign
- Drumming Fingers
- Thumb Up
- Thumb Down
- Zooming In With Full Hand
- Zooming In With Two Fingers
- Zooming Out With Full Hand
- Zooming Out With Two Fingers
- Swiping Down
- Swiping Left
- Swiping Right
- Swiping Up
- Sliding Two Fingers Down
- Sliding Two Fingers Left
- Sliding Two Fingers Right
- Sliding Two Fingers Up
- Pulling Hand In
- Pulling Two Fingers In

## Contact

For any problems, contact:

Ji Lin, jilin@mit.edu

Yaoyao Ding, yyding@mit.edu
