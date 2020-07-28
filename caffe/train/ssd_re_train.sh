#!/bin/bash

caffe_dir="/home/hena/caffe-ssd/caffe"

#create_list.sh
dataset_name="plate_locate"
version="v0.0.1"
data_list_dir="/work/hena/ocr/model/caffe/CLPR/det/v0.0.1/" #保存文件列表txt的路径
mapfile="${data_list_dir}labelmap_title_cls.prototxt"
solver_dir="${data_list_dir}solver.prototxt"
save_logs_dir="${data_list_dir}logs/${dataset_name}_${version}_$(date "+%Y%m%d%H%M").log"
weight=1

if [ $weight ]
then
  weight_dir="${data_list_dir}plate_locate_v001_iter_18800.caffemodel"
  solver_dir="${solver_dir} -weights ${weight_dir}"
fi




echo "========== Beging Train... =========="
echo "${caffe_dir}/build/tools/caffe train -gpu 0 -solver ${solver_dir} >& ${save_logs_dir}"
${caffe_dir}/build/tools/caffe train -gpu 0 -solver ${solver_dir} 2>&1 | tee ${save_logs_dir}