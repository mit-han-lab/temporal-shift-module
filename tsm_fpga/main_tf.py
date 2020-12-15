import numpy as np
import ntpath
import os
import re
import argparse
from typing import Tuple
import io
import time
import cv2
from PIL import Image, ImageOps
import tensorflow.compat.v1 as tf

QUANTIZE_RESULTS = "" # TODO: Set to path containing quantization results so post-quantization model can be loaded (see fpga_build directory)
SPLIT_GRAPH = True
JESTER_TEST = True
JESTER_PATH = ""# TODO: Set directly to jester video directory containing images (i.e. jester_data/vid_num/)
if JESTER_TEST:
    import tensorflow.contrib.decent_q

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    #frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame -= [0.485, 0.456, 0.406]
    frame /= [0.229, 0.224, 0.225]
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame

catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


n_still_frame = 0

def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2
    
    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]): #  and history[-2] == history[-3]):
            idx_ = history[-1]
    

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


WINDOW_NAME = 'Video Gesture Recognition'
def run(target='cuda', print_log=True, callback_fn=lambda output: None):
    """
    target: {'cuda', 'opencl', 'llvm [-target={LLVM_TARGET}]'
        * -target implies cross compilation. LLVM_TARGET is one of llvm cross compilation targets. See: https://clang.llvm.org/docs/CrossCompilation.html
    """
    if not JESTER_TEST:
        print("Open camera...")
        cap = cv2.VideoCapture(0)

        print(cap)

        # set a lower resolution for speed up
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)


    t = None
    index = 0
    #print("Build transformer...")
    #transform = get_transform()

    buffer = [
        np.zeros((1, 56, 56, 3)),
        np.zeros((1, 28, 28, 4)),
        np.zeros((1, 28, 28, 4)),
        np.zeros((1, 14, 14, 8)),
        np.zeros((1, 14, 14, 8)),
        np.zeros((1, 14, 14, 8)),
        np.zeros((1, 14, 14, 12)),
        np.zeros((1, 14, 14, 12)),
        np.zeros((1, 7, 7, 20)),
        np.zeros((1, 7, 7, 20))
    ]
    idx = 0
    history = [2,2]
    history_logit = []
    history_timing = []

    i_frame = -1

    print("Ready!")
    # Lock to 25 FPS
    frame_time = 0
    tf.reset_default_graph()

    sessions = []

    p_buffer = (
        np.zeros((1, 56, 56, 3)),
        np.zeros((1, 28, 28, 4)),
        np.zeros((1, 28, 28, 4)),
        np.zeros((1, 14, 14, 8)),
        np.zeros((1, 14, 14, 8)),
        np.zeros((1, 14, 14, 8)),
        np.zeros((1, 14, 14, 12)),
        np.zeros((1, 14, 14, 12)),
        np.zeros((1, 7, 7, 20)),
        np.zeros((1, 7, 7, 20))
    )
    p_in_img = np.zeros((1, 224, 224, 3))

    if SPLIT_GRAPH:
        for i in range(11):
            model_name = f"model_tf_split_{i}"
            model_path = f"{QUANTIZE_RESULTS}/quantize_results_{i}/quantize_eval_model.pb"
            graph_def = tf.GraphDef.FromString(open(model_path, "rb").read()) 
            graph = tf.import_graph_def(graph_def, name = "")
            sessions.append(tf.Session(graph=graph))
    else:
        graph_def = tf.GraphDef.FromString(open("./model_tf/model_tf.pb", "rb").read())
        graph = tf.import_graph_def(graph_def, name = "")
        sessions.append(tf.Session(graph=graph))


    while True:
        i_frame += 1
        if JESTER_TEST:
            imgs = sorted(os.listdir(JESTER_PATH))
            if i_frame >= len(imgs):
                break
            img = np.asarray(Image.open(os.path.join(JESTER_PATH, imgs[i_frame])))
            img = img[:,:,::-1]
        else:
            _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if time.time() - frame_time > 1/25:
            frame_time = time.time()
            t1 = time.time()
            img_tran = transform(np.array(Image.fromarray(img).convert('RGB')))

            feat = None
            if SPLIT_GRAPH:
                resid_in = None
                shift_concat_in = None
                for i,sess in enumerate(sessions):
                    out_node = None
                    feed_dict = {}
                    if i == 0:
                        out_node = "MobilenetV2/expanded_conv_shift/prev_conv_output:0"
                        feed_dict["in_img:0"] = img_tran#np.ones((1,224,224,3))
                        print("img: ", img[0][0], " = ", img.shape)
                        print("img_tran: ", img_tran[0][0][0], " = ", img_tran.shape)
                    else:
                        out_node = f"MobilenetV2/expanded_conv_shift_{i}/prev_conv_output:0"
                        layer_name = f"expanded_conv_shift_{i-1}" if i > 1 else f"expanded_conv_shift"
                        feed_dict[f"MobilenetV2/{layer_name}/input:0"] = resid_in
                        feed_dict[f"MobilenetV2/{layer_name}/shift_concat_input:0"] = shift_concat_in

                    if i == len(sessions) - 1:
                        out_node = "MobilenetV2/Logits/output:0"

                    output = sess.run(out_node, feed_dict)

                    if i == len(sessions) - 1:
                        print(output[0][0])
                    else:
                        print(output[0][0][0][0])

                    # Do shift
                    if i < len(sessions) - 1:
                        resid_in = output
                        c = int(output.shape[3])
                        x1, x2 = output[:, :, :, :c//8], output[:, :, :, c//8:]
                        print("x1: ", x1[0][0][0][0])
                        print("x2: ", x2[0][0][0][0])
                        shift_concat_in = np.concatenate((buffer[i], x2), axis=3)
                        buffer[i] = x1

                    if i == len(sessions) - 1:
                        feat = output
            else:
                out_nodes = ['MobilenetV2/Logits/output:0', 'MobilenetV2/expanded_conv_shift/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_1/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_2/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_3/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_4/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_5/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_6/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_7/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_8/shift_split_buffer_output:0', 'MobilenetV2/expanded_conv_shift_9/shift_split_buffer_output:0']
                feed_dict = {f"shift_buffer_{i}:0": buffer[i] for i in range(10)}
                feed_dict["in_img:0"] = img_tran
                outputs = sessions[0].run(out_nodes, feed_dict)
                feat, buffer = outputs[0], outputs[1:]

            print(feat)
            if SOFTMAX_THRES > 0:
                feat_np = feat.reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
                if print_log:
                    print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat, axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat, axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat)
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]


            idx, history = process_output(idx_, history)
            callback_fn(idx)

            t2 = time.time()
            if print_log:
                print(f"{index} {catigories[idx]}")


            current_time = t2 - t1

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)


        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    if not JESTER_TEST:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='cuda', nargs='?', help="Target for compilation. If set to 'llvm -target={LLVM_TARGET}', cross compilation is assumed. See run() function for additional details.")
    args = parser.parse_args()
    run(target=args.target)
