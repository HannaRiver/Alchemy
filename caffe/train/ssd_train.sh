#!/bin/bash

caffe_dir="/home/hena/caffe-ssd/caffe"

#create_list.sh
dataset_name="ocr_title"
version="v0.0.3"
data_root_dir="/work/hena/ocr/data/FinancialStatements/title"
data_list_dir="/work/hena/ocr/model/caffe/title_ssd/${version}/" #保存文件列表txt的路径
mapfile="${data_list_dir}model/labelmap_title_cls.prototxt"
solver_dir="${data_list_dir}model/solver.prototxt"
save_logs_dir="${data_list_dir}logs/${dataset_name}_${version}_$(date "+%Y%m%d%H%M").log"
weight=1

if [ $weight ]
then
  weight_dir="${data_list_dir}model/ocr_title_iter_110000.caffemodel"
  solver_dir="${solver_dir} -weights ${weight_dir}"
fi

for del_item in test.txt trainval.txt test_name_size.txt
do
  if [ -f $data_list_dir$del_item ]
  then
    rm -f $data_list_dir$del_item
  fi
done


redo=1
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=png --encoded"

echo "========== Beging create list... =========="
# for dataset in trainval test
for dataset in trainval
do
  echo "Create list for $dataset..."
  img_file="${data_list_dir}${dataset}_img.txt"
  label_file="${data_list_dir}${dataset}_label.txt"
  dst_file=${data_list_dir}$dataset.txt
  paste -d' ' $img_file $label_file >> $dst_file

  # if [ $dataset == "test" ]
  # then
  #   ${caffe_dir}/build/tools/get_image_size $data_root_dir/ $dst_file ${data_list_dir}${dataset}"_name_size.txt"
  # fi

  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
echo "========== create list done! =========="
echo "========== Beging create data... =========="
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
# for subset in test trainval
for subset in trainval
do
  python $caffe_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd ${data_root_dir} ${data_list_dir}$subset.txt ${data_list_dir}$db/$dataset_name"_"$subset"_"$db ${caffe_dir}/examples/$dataset_name
done
echo "========== create data done! =========="

echo "========== Beging Train... =========="
${caffe_dir}/build/tools/caffe train -gpu 0 -solver ${solver_dir} 2>&1 | tee ${save_logs_dir}