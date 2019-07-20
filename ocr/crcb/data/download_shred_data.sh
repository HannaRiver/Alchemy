#!/bin/bash

classify_dir='/data1/ocr/ocr_project/alchemy/ocr/crcb/data'   #原始和封装的脚本存放位置
# date='20180918'
date=$1
data_dir='/data1/ocr/data/org_print/download'   #数据下载到的保存位置

{
  cls_type='CA'
  save_dir=${data_dir}/${cls_type}
  python3 $classify_dir/download_shred_data.py --cls_type $cls_type --save_dir ${save_dir} --date ${date}
  echo "$date $cls_type done!"
} &


{
  cls_type='LA'
  save_dir=${data_dir}/${cls_type}
  python3 $classify_dir/download_shred_data.py --cls_type $cls_type --save_dir ${save_dir} --date ${date}
  echo "$date $cls_type done!"
} &


{
  cls_type='CD'
  save_dir=${data_dir}/${cls_type}
  python3 $classify_dir/download_shred_data.py --cls_type $cls_type --save_dir ${save_dir} --date ${date}
  echo "$date $cls_type done!"
} &

{
  cls_type='LD'
  save_dir=${data_dir}/${cls_type}
  python3 $classify_dir/download_shred_data.py --cls_type $cls_type --save_dir ${save_dir} --date ${date}
  echo "$date $cls_type done!"
} &


wait

echo "=================== Classify shrd data Done! ==================="
