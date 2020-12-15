import os
from PIL import Image
import numpy as np
import random
import pickle

IMAGENET_PATH = "/MEng/Data/ILSVRC2012_img_val/"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, .225]

CALIB_BASE_PATH=os.getenv("CALIB_BASE_PATH")
if CALIB_BASE_PATH is None:
    raise ValueError("Environment variable CALIB_BASE_PATH not set")

CALIB_MODEL_SPLIT=os.getenv("CALIB_MODEL_SPLIT")
if CALIB_MODEL_SPLIT is None:
    raise ValueError("Environment variable CALIB_MODEL_SPLIT not set")

quantize_info_path = os.path.join(CALIB_BASE_PATH, f"model_tf_split_{CALIB_MODEL_SPLIT}/quantize_info.txt")
input_info_path = os.path.join(CALIB_BASE_PATH, f"model_tf_split_{CALIB_MODEL_SPLIT}/inputs.pickle")

input_shapes = {}
with open(quantize_info_path) as f:
    lines = f.readlines()
    raw_input_names = []
    raw_input_shapes = []
    for i in range(len(lines)):
        if "--input_nodes" in lines[i]:
            raw_input_names = lines[i+1].rstrip()
        if "--input_shapes" in lines[i]:
            raw_input_shapes = lines[i+1].rstrip()

    raw_input_names = raw_input_names.split(",")
    raw_input_shapes = raw_input_shapes.split(":")
    raw_input_shapes = [[int(x) for x in shape.split(',')] for shape in raw_input_shapes]
    input_shapes = dict(zip(raw_input_names, raw_input_shapes))


input_data = {}
# shift_concat, resid
with open(input_info_path, 'rb') as f:
    input_data = pickle.load(f)

def input_fn(iter):
    #files = sorted(os.listdir(IMAGENET_PATH))
    #img = Image.open(os.path.join(IMAGENET_PATH,files[iter])).resize((224, 224))
    #img = np.array(img) / 255.0
    ##img = (img -  MEAN) / STD
    #img = np.transpose(img, axes=[2, 0, 1])
    #img = np.expand_dims(img, axis=0)
    #return {"input_node": img}
    inputs = {}
    for name,shape in input_shapes.items():
        if "/input" in name:
            inputs[name] = np.array(input_data[iter]["resid"])
            #inputs[name] = np.array(input_data["0"]["resid"])
        else:
            inputs[name] = np.array(input_data[iter]["shift_concat"])
            #inputs[name] = np.array(input_data["0"]["shift_concat"])

    #inputs = {name: np.random.rand(*shape) for name,shape in input_shapes.items()}

    return inputs
