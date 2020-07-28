#!/bin/bash

# caffe_dir="/home/hena/caffe-ssd/caffe"
caffe_dir="/home/hena/caffe-ocr/buildcmake/tools/caffe"

#create_list.sh
dataset_name="crcb_shred"
version="v0.0.1"

CAFFE_MODEL=/work/hena/ocr/model/caffe/HandWriting/isHW/
solver_dir="${CAFFE_MODEL}model/${dataset_name}_solver.prototxt"
save_logs_dir="${CAFFE_MODEL}logs/${dataset_name}_${version}_$(date "+%Y%m%d%H%M").log"

weight=true
if [ $weight ]
then
  weight_dir="${CAFFE_MODEL}model/${dataset_name}_cls.caffemodel"
  solver_dir="${solver_dir} -weights ${weight_dir}"
fi




echo "========== Beging Train... =========="

# ${caffe_dir}/build/tools/caffe train -gpu 0 -solver ${solver_dir} |tee ${save_logs_dir}
${caffe_dir} train -gpu 0 -solver ${solver_dir} 2>&1 |tee ${save_logs_dir}