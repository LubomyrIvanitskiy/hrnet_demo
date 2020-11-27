#!/bin/bash
# Download and setup HRNet
HRNET=hrnet
git clone https://github.com/stefanopini/simple-HRNet.git $HRNET
python -m pip install -r $HRNET/requirements.txt
wget -nc -O weights https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
YOLO=$HRNET/models/detectors/yolo
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3 $YOLO
python -m pip install -r $YOLO/requirements.txt
chmod 755 $YOLO/weights/download_weights.sh
sh $YOLO/weights/download_weights.sh
mv yolov3.weights $YOLO/weights
mv yolov3-tiny.weights $YOLO/weights
rm -r weights
mkdir weights
wget -nc -O weights/pose_hrnet_w48_384x288.pth https://www.dropbox.com/s/pjd3bzxajmhksct/pose_hrnet_w48_384x288.pth?dl=0