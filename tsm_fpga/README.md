# TSM Deployed to FPGA [[Demo]](https://www.youtube.com/watch?v=dy6-uPzg86c)

We deploy TSM to FPGA using the Vitis-AI framework. To do so, we generate a tensorflow implementation of the TSM model, and pipeline the network such that all shift operations are isolated. This allows deployment of the Shift operation to CPU and the remaining operations to the Vitis-AI DPU IP.

We must take additional steps to deploy this pipelined model. First, isolating the shift-operations results in a number of seperate DPU kernels for the seperate portions of the network (11 for MobileNetV2 TSM). These kernels must be quantized to int8 and compiled for DPU seperately.

To quantize the split model, we dump intermediate activations from the unsplit implementation at the locations of DPU kernel inputs. These inputs are then used as input to the Vitis-AI quantizer. Once quantized, the resulting splits of the model can be compiled into the final demo executable.

![split-mbv2](https://github.com/mit-han-lab/temporal-shift-module/raw/master/tsm_fpga/images/split_mobilenetv2_bottleneck.png)

## FPGA Setup

To build the FPGA project, ensure you have initialized to tensorflow-slim submodule (git submodule update --init --recursive).

This was tested with the ZCU104 MPSOC DPU TRD in the Vitis-AI repository and the Ultra96V2 Avnet 2020.1 beta branch (https://github.com/Avnet/vitis/tree/2020.1) (See the following guide for additional build instructions https://www.hackster.io/AlbertaBeef/vitis-ai-1-1-flow-for-avnet-vitis-platforms-part-2-f18be4)

### 1) Dump Split TF Models
The `mobilenet_v2_tfslim.py` is the primary scripts to build the online-TSM model for FPGA. To generate the split model set `SPLIT_MODEL`,`SPLIT_EXPORT`,and EXPORT to True at the top of the files. After running the script, you will see the split model dumped to the `model_tf_split_*` directories.

### 2) Dump Quantization Inputs
To gather quantization information, one must run the unsplit models. To do so ensure you set to quantize data paths at the TODOs at the top of the files. Then set `SPLIT_MODEL`,`SPLIT_EXPORT`, and EXPORT to False. Then set the corresponding `QUANTIZE_*` flag and `DUMP_QUANTIZE` flag to True to enable quantization.

### 3) Quantize & Compile DPU Kernels
Once quantization data is generated (see `inputs.pickle` and `quantize_info.txt` under the `model_tf_split_export/*` directories), one can move to the `fpga_build` to quantize and compile each split of the model. 

Update `compile_split.sh` to use the correct target architecture variable. Use the `quantize_split.sh` and `compile_split.sh` files to launch `vai_q_tensorflow` and `vai_c_tensorflow` respectively (from within the docker container).

### 4) Compile demo executable
Once model quantization is complete, in the `fpga_build/model_tf_split` directory one can run "make `ultra96v2.tsm_online`" or "make `zcu104.tsm_online` to generate the demo executable for a given target from the src files and generated DPU kernels.

## Ultra96V2 Online-TSM Jester Demo

On Ultra96V2 we achieve an inference throughput of 37 FPS with a power consumpstion of 10.6W.
A recording of this demo along with project description can be found at [[ultra96v2-demo]](https://www.youtube.com/watch?v=dy6-uPzg86c).

