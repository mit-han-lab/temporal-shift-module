#include <iostream>
#include <string>
#include <tuple>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <queue>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <dnndk/n2cube.h>

#include "helper.h"

const bool JESTER_TEST = false;
std::string JESTER_PATH = "./jester_test/9223";

const bool HEADLESS = false;
std::string WINDOW_NAME = "TSM on FPGA";

const bool USE_SOFTMAX = false;

const bool VID_STAT_DISPLAY = true;

// mode for dpuCreateTask (T_MODE_NORMAL, T_MODE_PROF, T_MODE_DEBUG
#define TASK_MODE T_MODE_NORMAL
#define NUM_KERNELS 11
bool DEVICE_OPEN = false;

typedef int8_t wquant;
typedef int8_t aquant;

using std::chrono::high_resolution_clock;
template <typename T>
auto to_ms(T const& duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void fail(std::string err) {
    std::cerr << err << std::endl;

    if (DEVICE_OPEN)
        dpuClose();

    exit(1);
}

struct ShiftBufferPool {
    aquant b0[56][56][3] = {0};
    aquant b1[28][28][4] = {0};
    aquant b2[28][28][4] = {0};
    aquant b3[14][14][8] = {0};
    aquant b4[14][14][8] = {0};
    aquant b5[14][14][8] = {0};
    aquant b6[14][14][12] = {0};
    aquant b7[14][14][12] = {0};
    aquant b8[7][7][20] = {0};
    aquant b9[7][7][20] = {0};

    aquant* operator[](int i) {
        switch(i) {
            case 0: return &b0[0][0][0];
            case 1: return &b1[0][0][0];
            case 2: return &b2[0][0][0];
            case 3: return &b3[0][0][0];
            case 4: return &b4[0][0][0];
            case 5: return &b5[0][0][0];
            case 6: return &b6[0][0][0];
            case 7: return &b7[0][0][0];
            case 8: return &b8[0][0][0];
            case 9: return &b9[0][0][0];
            default: fail("Invalid shift buffer");
        }
    }
};

struct Node {
    std::string name; // "" marks invalid node
    int shape[3]; // HWC

    size_t size() const { return sizeof(aquant)*shape[0]*shape[1]*shape[2]; }
};
struct TSMSplit {
    std::string kernelName;
    Node resid_input_node;
    Node shift_input_node;
    Node output_node;

    aquant* virt_in = nullptr;
    aquant* phys_in = nullptr;
    aquant* virt_out = nullptr;
    aquant* phys_out = nullptr;

    void* inHandle = nullptr;
    void* outHandle = nullptr;

    aquant* residAddr() { return virt_in; }
    aquant* shiftAddr() { return virt_in + resid_input_node.size(); }
    aquant* outAddr()   { return virt_out; }

    size_t inSize() { return resid_input_node.size() + shift_input_node.size(); }
    size_t outSize() { return output_node.size(); }
};

// list of kernel information:
// {Kernel Name,
//      residual tensor {name, shape}
//      shifted tensor {name, shape}
//      output tensor {name, shape}
TSMSplit splitInfo[NUM_KERNELS] = {
    {"tsm_mobilenet_v2_0",
        {"", {0,0,0}},
        {"in_img:0", {224,224,3}},
        {"MobilenetV2_expanded_conv_1_project_Conv2D:0", {56,56,24}}},
    {"tsm_mobilenet_v2_1",
        // Residual Input
        {"MobilenetV2_expanded_conv_shift_input:0", {56,56,24}},
        // Shifted input
        {"MobilenetV2_expanded_conv_shift_shift_concat_input:0", {56,56,24}},
        // Output
        {"MobilenetV2_expanded_conv_2_project_Conv2D:0", {28,28,32}}},
    {"tsm_mobilenet_v2_2",
        {"MobilenetV2_expanded_conv_shift_1_input:0",               {28,28,32}},
        {"MobilenetV2_expanded_conv_shift_1_shift_concat_input:0",  {28,28,32}},
        {"MobilenetV2_expanded_conv_shift_1_add:0",                 {28,28,32}}},
    {"tsm_mobilenet_v2_3",
        {"MobilenetV2_expanded_conv_shift_2_input:0",               {28,28,32}},
        {"MobilenetV2_expanded_conv_shift_2_shift_concat_input:0",  {28,28,32}},
        {"MobilenetV2_expanded_conv_3_project_Conv2D:0",            {14,14,64}}},
    {"tsm_mobilenet_v2_4",
        {"MobilenetV2_expanded_conv_shift_3_input:0",               {14,14,64}},
        {"MobilenetV2_expanded_conv_shift_3_shift_concat_input:0",  {14,14,64}},
        {"MobilenetV2_expanded_conv_shift_3_add:0",                 {14,14,64}}},
    {"tsm_mobilenet_v2_5",
        {"MobilenetV2_expanded_conv_shift_4_input:0",               {14,14,64}},
        {"MobilenetV2_expanded_conv_shift_4_shift_concat_input:0",  {14,14,64}},
        {"MobilenetV2_expanded_conv_shift_4_add:0",                 {14,14,64}}},
    {"tsm_mobilenet_v2_6",
        {"MobilenetV2_expanded_conv_shift_5_input:0",               {14,14,64}},
        {"MobilenetV2_expanded_conv_shift_5_shift_concat_input:0",  {14,14,64}},
        {"MobilenetV2_expanded_conv_4_project_Conv2D:0",            {14,14,96}}},
    {"tsm_mobilenet_v2_7",
        {"MobilenetV2_expanded_conv_shift_6_input:0",               {14,14,96}},
        {"MobilenetV2_expanded_conv_shift_6_shift_concat_input:0",  {14,14,96}},
        {"MobilenetV2_expanded_conv_shift_6_add:0",                 {14,14,96}}},
    {"tsm_mobilenet_v2_8",
        {"MobilenetV2_expanded_conv_shift_7_input:0",               {14,14,96}},
        {"MobilenetV2_expanded_conv_shift_7_shift_concat_input:0",  {14,14,96}},
        {"MobilenetV2_expanded_conv_5_project_Conv2D:0",            {7,7,160}}},
    {"tsm_mobilenet_v2_9",
        {"MobilenetV2_expanded_conv_shift_8_input:0",               {7,7,160}},
        {"MobilenetV2_expanded_conv_shift_8_shift_concat_input:0",  {7,7,160}},
        {"MobilenetV2_expanded_conv_shift_8_add:0",                 {7,7,160}}},
    {"tsm_mobilenet_v2_10",
        {"MobilenetV2_expanded_conv_shift_9_input:0",               {7,7,160}},
        {"MobilenetV2_expanded_conv_shift_9_shift_concat_input:0",  {7,7,160}},
        {"MobilenetV2_Logits_Linear_MatMul:0",                      {1,1,27}}}
};


void allocBuffers(DPUTask* tasks[NUM_KERNELS]) {
    for (int i = 0; i < NUM_KERNELS; i++) {
        // TODO: Implement custom alloc
        // Allocate input buffers {residual, shifted}
        splitInfo[i].inHandle = dpuAllocMem(splitInfo[i].inSize(),
                (int8_t*&)splitInfo[i].virt_in, (int8_t*&)splitInfo[i].phys_in);
        splitInfo[i].outHandle = dpuAllocMem(splitInfo[i].outSize(),
                (int8_t*&)splitInfo[i].virt_out, (int8_t*&)splitInfo[i].phys_out);

        if(!splitInfo[i].inHandle)
            fail("Failed to allocated split input buffer");
        if(!splitInfo[i].outHandle)
            fail("Failed to allocated split output buffer");

        dpuBindInputTensorBaseAddress(tasks[i],
                (int8_t*&)splitInfo[i].virt_in, (int8_t*&)splitInfo[i].phys_in);
        dpuBindOutputTensorBaseAddress(tasks[i],
                (int8_t*&)splitInfo[i].virt_out, (int8_t*&)splitInfo[i].phys_out);
    }
}

void freeBuffers() {
    for (int i = 0; i < NUM_KERNELS; i++) {
        dpuFreeMem(splitInfo[i].inHandle);
        dpuFreeMem(splitInfo[i].outHandle);
    }
}

void rescale_input(int split_num, float scale) {
    const Node& input_node = splitInfo[split_num - 1].output_node;
    aquant* input_data = splitInfo[split_num - 1].outAddr();

    for (int i = 0; i < input_node.shape[0]; i++) {
        for (int j = 0; j < input_node.shape[1]; j++) {
            for (int k = 0; k < input_node.shape[2]; k++) {
                input_data[i*input_node.shape[1]*input_node.shape[2]
                    + j*input_node.shape[2] + k] *= scale;
            }
        }
    }
}

void doShift(int split_num, ShiftBufferPool& shift_buffers) {
    assert(split_num > 0);

    const Node& input_node = splitInfo[split_num - 1].output_node;
    aquant* input_data = splitInfo[split_num - 1].outAddr();

    const Node& output_node = splitInfo[split_num].shift_input_node;
    aquant* output_data = splitInfo[split_num].shiftAddr();

    // First task is only input, no shift node
    aquant* buffer = shift_buffers[split_num - 1];

    int c = input_node.shape[2] / 8;

    for (int i = 0; i < input_node.shape[0]; i++) {
        for (int j = 0; j < input_node.shape[1]; j++) {
            aquant* in = input_data + i*input_node.shape[1]*input_node.shape[2]
                                    + j*input_node.shape[2];

            aquant* out = output_data + i*input_node.shape[1]*input_node.shape[2]
                                      + j*input_node.shape[2];
            aquant* shift = buffer + i*input_node.shape[1]*c
                                   + j*c;
            //printf("(%d, %d)\n", i, j);
            memcpy(out, shift, c*sizeof(aquant));
            memcpy(out + c, in + c, (input_node.shape[2] - c)*sizeof(aquant));
            memcpy(shift, in, c*sizeof(aquant));
        }
    }

    // TODO: Merge residual and input split memory regions
    memcpy(splitInfo[split_num].residAddr(), input_data, input_node.size());
}

void runTSMSerial(DPUTask* tasks[NUM_KERNELS], ShiftBufferPool& shift_buffers) {
    float prev_out_scale;
    for (int i = 0; i < NUM_KERNELS; i++) {

        //// SCALING
        float out_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[i], splitInfo[i].output_node.name.c_str()));
        float shift_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[i], splitInfo[i].shift_input_node.name.c_str()));

        float resid_scale = 0;

        if (i > 0) {
            resid_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[i], splitInfo[i].resid_input_node.name.c_str()));

            // In all scales a costant scale is applied across all inputs
            assert(shift_scale == resid_scale);

            // Rescale input if previous output scaling doesn't cancel current input scaling
            // i.e. prev_out_scale * in_scale != 1
            float rescale = prev_out_scale * shift_scale;
            if (rescale != 1) {
                //printf("RESCALE (%f)\n", rescale);
                rescale_input(i, rescale);
            }


            //// SHIFTING
            auto t1 = high_resolution_clock::now();
            doShift(i, shift_buffers);
            auto t2 = high_resolution_clock::now();
            //printf("shift %d = %d\n", i, (t2-t1).count());
        }


        //// TASK RUNNING
        dpuSyncMemToDev(splitInfo[i].inHandle, 0, splitInfo[i].inSize());
        if (!VID_STAT_DISPLAY)
            printf("Running task [%d]\n", i);
        dpuRunTask(tasks[i]);

        dpuSyncDevToMem(splitInfo[i].outHandle, 0, splitInfo[i].outSize());

        prev_out_scale = out_scale;
    }
}

std::string categories[27] = {
    "Doing other things",  // 0
    "Drumming Fingers",  // 1
    "No gesture",  // 2
    "Pulling Hand In",  // 3
    "Pulling Two Fingers In",  // 4
    "Pushing Hand Away",  // 5
    "Pushing Two Fingers Away",  // 6
    "Rolling Hand Backward",  // 7
    "Rolling Hand Forward",  // 8
    "Shaking Hand",  // 9
    "Sliding Two Fingers Down",  // 10
    "Sliding Two Fingers Left",  // 11
    "Sliding Two Fingers Right",  // 12
    "Sliding Two Fingers Up",  // 13
    "Stop Sign",  // 14
    "Swiping Down",  // 15
    "Swiping Left",  // 16
    "Swiping Right",  // 17
    "Swiping Up",  // 18
    "Thumb Down",  // 19
    "Thumb Up",  // 20
    "Turning Hand Clockwise",  // 21
    "Turning Hand Counterclockwise",  // 22
    "Zooming In With Full Hand",  // 23
    "Zooming In With Two Fingers",  // 24
    "Zooming Out With Full Hand",  // 25
    "Zooming Out With Two Fingers"  // 26
};

int processOutput(int raw_gesture, float features[27]) {
    const int HISTORY_FEAT_LEN = 12;
    const int HISTORY_LEN = 20;

    static int i_logit = 0;
    static std::vector<std::array<float, 27>> history_feat(HISTORY_FEAT_LEN, std::array<float, 27>{});
    static std::deque<int> history = {2, 2};

    // Copy round of features into running feature history array
    std::copy_n(features, 27, history_feat[i_logit].data());
    i_logit++;
    if (++i_logit >= 12)
        i_logit = 0;

    // Get current gesture across length of feature history
    std::array<float, 27> sums{};
    for (int i = 0; i < HISTORY_FEAT_LEN; i++) {
        for (int j = 0; j < 27; j++) {
            sums[j] += history_feat[i][j];
        }
    }

    int gesture = std::distance(sums.begin(), std::max_element(sums.begin(), sums.end()));

	if (gesture == 0)
		gesture = 2;

    // Apply history smoothing
    if (gesture != *history.rbegin() && *history.rbegin() != *(history.rbegin() + 1))
        gesture = *history.rbegin();

    history.push_back(gesture);
    if (history.size() > HISTORY_LEN)
        history.pop_front();

    return *history.rbegin();
}

int run(cv::VideoCapture& cap) {
    DPUKernel* kernels[NUM_KERNELS];
    DPUTask*   tasks[NUM_KERNELS];

    if (dpuOpen() != 0)
        fail("Error opening DPU device");

    DEVICE_OPEN = true;

    for (int i = 0; i < NUM_KERNELS; i++) {
        kernels[i] = dpuLoadKernel(splitInfo[i].kernelName.c_str());
        tasks[i] = dpuCreateTask(kernels[i], TASK_MODE);
    }

    allocBuffers(tasks);

    cv::Mat frame, frame_rgb;

    if (!HEADLESS) {
        cv::namedWindow(WINDOW_NAME);
        cv::setWindowTitle(WINDOW_NAME, WINDOW_NAME);
    }

    std::cout << "Running...\n";

    auto cv_type = CV_8UC3;
    if (typeid(aquant) == typeid(uint16_t)) {
        cv_type = CV_16UC3;
    } else if(typeid(aquant) == typeid(uint32_t)) {
        cv_type = CV_32SC1;
    }
    cv::Mat in_img(splitInfo[0].shift_input_node.shape[0],
                   splitInfo[0].shift_input_node.shape[1],
                   cv_type, splitInfo[0].shiftAddr());

    std::unique_ptr<ShiftBufferPool> shift_buffers = std::make_unique<ShiftBufferPool>();

    float* softmax = new float[27];
    float out_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[NUM_KERNELS-1], splitInfo[NUM_KERNELS-1].output_node.name.c_str()));
    float in_scale = dpuGetTensorScale(dpuGetBoundaryIOTensor(tasks[0], splitInfo[0].shift_input_node.name.c_str()));

    printf("inscale: %f, outscale: %f\n", in_scale, out_scale);

    std::vector<std::string> jester_imgs;
    if (JESTER_TEST) {
        jester_imgs = listDir(JESTER_PATH);
    	std::cout << jester_imgs[0] << std::endl;
	}

    auto t_lastframe = high_resolution_clock::now();
    int frame_num = -1;
    for (;;) {
        if (to_ms(high_resolution_clock::now() - t_lastframe) > 1000/15) {
            frame_num++;
            t_lastframe = high_resolution_clock::now();

            auto t_preframe = high_resolution_clock::now();
            if (JESTER_TEST) {
                if (frame_num >= jester_imgs.size())
                    break;
                frame = cv::imread(jester_imgs[frame_num]);
            } else {
                cap.read(frame);
            }
            auto t_postframe = high_resolution_clock::now();
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            frame.convertTo(frame_rgb, CV_32FC3, 1/255.0);
            int new_w = 256;
            int new_h = 256;
            if (frame.cols > frame.rows) {
                new_w = 256*frame.cols/frame.rows;
            } else {
                new_h = 256*frame.rows/frame.cols;
            }
            cv::resize(frame_rgb, frame_rgb, cv::Size(new_w, new_h));
            frame_rgb = frame_rgb(cv::Rect((new_w - 224)/2, (new_h - 224)/2, 224, 224)).clone();

            // Normalize:
            // mean = (0.485, 0.456, 0.406) = (124, i27, 104)
            // std = (0.229, 0.224, 0.225)
            frame_rgb -= cv::Scalar(0.485,0.456, 0.406);
            cv::divide(frame_rgb, cv::Scalar(0.229, 0.224, 0.225), frame_rgb);
            frame_rgb.convertTo(in_img, cv_type, in_scale);
            auto t_postprocess = high_resolution_clock::now();

            runTSMSerial(tasks, *shift_buffers);
            aquant* features = splitInfo[NUM_KERNELS-1].outAddr();

			float scaled_features[27];
			for (int i = 0; i < 27; i++) {
				scaled_features[i] = features[i]*out_scale;
			}

            int max = 0;
			float max_val;
            if (USE_SOFTMAX) {
                dpuRunSoftmax(features, softmax, 27, 1, out_scale);
                max_val = softmax[0];
                for (int i = 1; i < 27; i++ ) {
                    if (softmax[i] > max_val) {
                        max = i;
                        max_val = softmax[i];
                    }
                }
            } else {
				max_val = scaled_features[0];
                for (int i = 1; i < 27; i++ ) {
                    if (scaled_features[i] > max_val) {
                        max = i;
                        max_val = scaled_features[i];
                    }
                }
            }
           
            int gesture = processOutput(max, scaled_features);
            if (!VID_STAT_DISPLAY)
                printf("Gesture: %s (%d); (%d, %f)\n", categories[gesture].c_str(), gesture, max, max_val);

            auto t_postrun = high_resolution_clock::now();

            if (!HEADLESS) {
                cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
                cv::flip(frame, frame, 1);

                int key;
                if (VID_STAT_DISPLAY) {
                    float fps = 1000 / to_ms(t_postrun - t_preframe);

                    char predict_str[64];
                    cv::Mat txt(cv::Size(frame.cols, 24), CV_8UC3, cv::Scalar(0));
                    sprintf(predict_str, "Prediction: %s | FPS: %.1f", categories[gesture].c_str(), fps);
                    cv::putText(txt, predict_str, cv::Point(5, txt.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);

                    cv::Mat display_frame;
                    cv::vconcat(frame, txt, display_frame);
                    cv::imshow(WINDOW_NAME, display_frame);
                    key = cv::waitKey(80);
                } else {
                    key = cv::waitKey(1);
                    cv::imshow("TSM on FPGA", frame);
                }
                if (key == 0x71)
                    break;
            }
            auto t_postshow = high_resolution_clock::now();

            std::uint64_t d_framegrab = std::chrono::duration_cast<std::chrono::microseconds>(t_postframe - t_preframe).count();
            std::uint64_t d_frameprocess = std::chrono::duration_cast<std::chrono::microseconds>(t_postprocess - t_postframe).count();
            std::uint64_t d_run = std::chrono::duration_cast<std::chrono::microseconds>(t_postrun - t_postprocess).count();
            std::uint64_t d_show = std::chrono::duration_cast<std::chrono::microseconds>(t_postshow - t_postrun).count();
            std::uint64_t d_total = std::chrono::duration_cast<std::chrono::microseconds>(t_postshow - t_preframe).count();
            float fps = 1000 / to_ms(t_postrun - t_preframe);

            if (!VID_STAT_DISPLAY) {
                printf("framegrab: %lu; frameprocess: %lu; run: %lu; show: %lu; total: %lu\n", d_framegrab, d_frameprocess, d_run, d_show, d_total);
                printf("FPS (frameprocess -> run): %f\n", fps);
            }
        }
    }

    delete softmax;
    freeBuffers();

    bool err = false;
    for (int i = 0; i < NUM_KERNELS; i++) {
        int t = dpuDestroyTask(tasks[i]);
        int d = dpuDestroyKernel(kernels[i]);
        if (d != 0 || t != 0)
            err = true;
    }
    if (err)
        fail("Error destroying kernel/task");


    if (dpuClose() != 0)
        fail("Error closing DPU device");

    return 0;
}

int main() {
    cv::VideoCapture cap;
    cap.open(0, cv::CAP_ANY);

    if (!cap.isOpened())
        fail("Error opening camera");

    // Set 320 x 240 crop
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Capture Resolution: " << w << "x" << h << "\n";


    int ret = run(cap);

    cap.release();
    cv::destroyAllWindows();

    return ret;
}
