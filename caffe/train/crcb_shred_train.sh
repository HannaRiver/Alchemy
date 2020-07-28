#!/bin/bash

caffe_dir1="/home/hena/caffe-ssd/caffe"
caffe_dir="/home/hena/caffe-ocr/buildcmake/tools/caffe"

dataset_name="crcb_shred"
version="v0.0.1"

# binary classifier lmdb date of crcb shred 
EXAMPLE=/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/bicls/lmdb
TOOLS=${caffe_dir1}/build/tools

DATA_ROOT=/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/
DATA=/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/bicls/

CAFFE_MODEL=/work/hena/ocr/model/caffe/HandWriting/isHW/
solver_dir="${CAFFE_MODEL}model/${dataset_name}_solver.prototxt"
save_logs_dir="${CAFFE_MODEL}logs/${dataset_name}_${version}_$(date "+%Y%m%d%H%M").log"

if [ ! -d "${EXAMPLE}" ];then
  echo "Warning: save lmdb data root is not existes: ${EXAMPLE}"
  mkdir ${EXAMPLE}
fi

if [ ! -d "$DATA_ROOT" ];then
  echo "Error: DATA_ROOT is not valid path: $DATA_ROOT"
  exit 1
fi

for del_item in $EXAMPLE/crcb_192x512_train_lmdb $EXAMPLE/crcb_192x512_val_lmdb
do
  if [ -d ${del_item} ]
  then
    rm -r ${del_item}
  fi
done

weight=1

if [ $weight ]
then
  weight_dir="${CAFFE_MODEL}model/${dataset_name}_cls.caffemodel"
  solver_dir="${solver_dir} -weights ${weight_dir}"
fi

RESIZE=true

if $RESIZE;
then
  RESIZE_HEIGHT=192
  RESIZE_WIDTH=512
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=true \
    $DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/crcb_192x512_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=true \
    $DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/crcb_192x512_val_lmdb

echo "Creating lmdb Done."

echo "========== Beging Train... =========="

# ${caffe_dir}/build/tools/caffe train -gpu 0 -solver ${solver_dir} >& ${save_logs_dir}
${caffe_dir} train -gpu 0 -solver ${solver_dir} 2>&1 |tee ${save_logs_dir}