# Download the pre-trained TSM models

echo 'Downloading TSN resnet50 on Kinetics...'
wget -P pretrained https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth

echo 'Downloading TSM resnet50 8-frame on Kinetics...'
wget -P pretrained https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth

echo 'Downloading TSM resnet50 16-frame on Kinetics...'
wget -P pretrained https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth

echo 'Downloading TSM resnet50 8-frame dense-sampling on Kinetics...'
wget -P pretrained https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth

echo 'Downloading Non-local TSM resnet50 8-frame dense-sampling on Kinetics...'
wget -P pretrained https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth