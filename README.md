# Temporal Shift Module for Efficient Video Understanding

We release the PyTorch code of the [Temporal Shift Module](https://arxiv.org/abs/1811.08383).

![framework](https://hanlab.mit.edu/projects/tsm/external/TSM-module.png)

### Reference

If you find our paper and repo useful, please cite our paper. Thanks!

```
@article{lin2018temporal,
    title={Temporal Shift Module for Efficient Video Understanding},
    author={Lin, Ji and Gan, Chuang and Han, Song},
    journal={arXiv preprint arXiv:1811.08383},
    year={2018}
}  
```

### Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.0
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

### Data Preparation

We need to first extract videos into frames for fast reading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing.

We have successfully trained on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2), [Jester](https://20bn.com/datasets/jester) datasets with this codebase. Basically, the processing of video data can be summarized into 3 steps:

- Extract frames from videos (refer to [tools/vid2img_kinetics.py](tools/vid2img_kinetics.py) for Kinetics example and [tools/vid2img_sthv2.py](tools/vid2img_sthv2.py) for Something-Something-V2 example)
- Generate annotations needed for dataloader (refer to [tools/gen_label_kinetics.py](tools/gen_label_kinetics.py) for Kinetics example, [tools/gen_label_sthv1.py](tools/gen_label_sthv1.py) for Something-Something-V1 example, and [tools/gen_label_sthv2.py](tools/gen_label_sthv2.py) for Something-Something-V2 example)
- Add the information to [ops/dataset_configs.py](ops/dataset_configs.py)

### Code

This code is based on the [TSN](https://github.com/yjxiong/temporal-segment-networks) codebase. The core code to implement the Temporal Shift Module is [ops/temporal_shift.py](ops/temporal_shift.py). It is a plug-and-play module to enable temporal reasoning, at the cost of *zero parameters* and *zero FLOPs*.

Here we provide a naive implementation of TSM. It can be implemented with just several lines of code:

```python
# shape of x: [N, T, C, H, W] 
out = torch.zeros_like(x)
fold = c // fold_div
out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
return out
```

Note that the naive implementation involves large data copying and increases memory consumption during training. It is suggested to use the **in-place** version of TSM to improve speed (**TODO**).

### Kinetics Pretrained Models

Training on Kinetics is computationally expensive. Here we provide the pretrained models on Kinetics for fine-tuning. To get the pretrained model, run from the root folder:

```
bash pretrained/download.sh
```

It will download the models into `pretrained` folder.

#### Dense Sampling Models

In the current version of our paper, we reported the results of TSM trained and tested with **I3D dense sampling** (Table 1&4, 8-frame and 16-frame), using the same training and testing hyper-parameters as in [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) paper to directly compare with I3D. Here we provide the 8-frame version checkpoint `TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth` that achieves 74.1% Kinetics accuracy. We also provide a model trained with **Non-local module** `TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth` to form NL TSM.

We compare the I3D performance reported in Non-local paper:

| method          | n-frame      | Kinetics Acc. |
| --------------- | ------------ | ------------- |
| I3D-ResNet50    | 32 * 10clips | 73.3%         |
| TSM-ResNet50    | 8 * 10clips  | **74.1%**     |
| NL I3D-ResNet50 | 32 * 10clips | 74.9%         |
| NL TSM-ResNet50 | 8 * 10clips  | **75.6%**     |

TSM outperforms I3D under the same dense sampling protocol. NL TSM model also achieves better performance than NL I3D model. Non-local module itself improves the accuracy by 1.5%.

#### Unifrom Sampling Models

We also provide the checkpoints of TSN and TSM models using **uniform sampled frames** as in [Temporal Segment Networks](<https://arxiv.org/abs/1608.00859>) paper, which is very useful for fine-tuning on other datasets. We provide the pretrained ResNet-50 for TSN and our TSM (8 and 16 frames), including `TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth, TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth, TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth`.

The performance on Kinetics is measured as (using only 1 clip):

| method     | n-frame | acc (1-crop) | acc (10-crop) |
| ---------- | ------- | ------------ | ------------- |
| TSN        | 8       | 68.8%        | 69.9%         |
| TSM (ours) | 8       | 71.2%        | 72.8%         |
| TSN        | 16      | 69.4%        | 70.2%         |
| TSM (ours) | 16      | **72.6%**    | **73.7%**     |

Our TSM module improves consistently over the TSN baseline.

### Testing 

For example, to test the downloaded pretrained models on Kinetics, you can run `scripts/test_tsm_kinetics_rgb_8f.sh`. The scripts will test both TSN and TSM on 8-frame setting by running:

```bash
# test TSN
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64

# test TSM
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64
```

Change to `--test_crops=10` for 10-crop evaluation. With the above scripts, you should get around 68.8% and 71.2% results respectively.

To get the Kinetics performance of our dense sampling model under Non-local protocol, run:

```bash
# test TSN using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res

# test TSM using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res

# test NL TSM using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res
```

You should get around 70.6%, 74.1%, 75.6% top-1 accuracy, as shown in Table 1.

We provide the **log files** of above testing examples in folder `logs`. For other datasets and trained models, refer to the code for details.

### Training 

We provided several examples to train TSM with this repo:

- To train on Kinetics from ImageNet pretrained models, you can run `scripts/train_tsm_kinetics_rgb_8f.sh`, which contains:

  ```bash
  # You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
  python main.py kinetics RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.02 --wd 5e-4 --lr_steps 20 40 --epochs 50 \
       --batch-size 128 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --npb
  ```

  You should get `TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth` as downloaded above. Notice that you should scale up the learning rate with batch size. For example, if you use a batch size of 256 you should set learning rate to 0.04.

- After getting the Kinetics pretrained models, we can fine-tune on other datasets using the Kinetics pretrained models. For example, we can fine-tune 8-frame Kinetics pre-trained model on UCF-101 dataset using **uniform sampling** by running:

  ```
  python main.py ucf101 RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
       --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres \
       --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
  ```

- Similarly, you can fine-tune the 16-frame model on Something-Something-V1 dataset using **uniform sampling** by running:

  ```bash
  python main.py something RGB \
       --arch resnet50 --num_segments 16 \
       --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
       --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres \
       --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth
  ```

### Live Demo on NVIDIA Jetson TX2

We have build an online hand gesture recognition demo using our TSM. The model is built with MobileNetV2 backbone and trained on Jester dataset. 

- Recorded video of the live demo [[link]](https://hanlab.mit.edu/projects/tsm/#live_demo)
- Code of the live demo on Jeston TX2: [TODO]
