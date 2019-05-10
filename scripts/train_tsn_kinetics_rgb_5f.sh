# You should get TSM_kinetics_RGB_resnet50_avg_segment5_e50
# Notice that for TSN 2D baseline, it is recommended to train using 5 segments and test with more segments to avoid overfitting

python main.py kinetics RGB \
     --arch resnet50 --num_segments 5 \
     --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 128 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --npb