#!/bin/bash

data_dir='/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/Date'
tool_dir='/work/hena/scripts'
date_type1='凭证日期'
date_type2='支票签发日期'

# ssd的参数
# labelmap_file='/work/tmp/SignatureRec/model/CapitalMoneyRecog/labelmap.prototxt'
# mean_value=[185, 185, 185]
# model_def='/work/tmp/SignatureRec/model/CapitalMoneyRecog/CapitalMoneyRecogSSD.prototxt'
# image_resize=[160, 480]
# model_weights='/work/tmp/SignatureRec/model/CapitalMoneyRecog/CapitalMoneyRecogSSD.caffemodel'

# 处理凭证日期
# for date in '20180731' '20180802' '20180803' '20180808' '20180813' '20180814' '20180816' '20180827' '20180828' '20180829'
# do
#   # 统一修改确认后标注的名字 --> 统一变成相同的大写日期
#   data_path1="$data_dir/$date_type1/$date"
#   # mv "$data_path1/chn_roi" "$data_path1/roi"
#   echo $data_path1
#   python3 "$tool_dir/ocr/crcb/cdata_rename.py" $data_path1

#   # 生成临时的rename_mark及对应的label list
#   python3 "$tool_dir/ocr/crcb/crcb_copy.py" $data_dir $date_type1 $date
# done

# 处理支票签发日期
for date in '20180731' '20180802' '20180803' '20180808' '20180813' '20180814' '20180816' '20180827' '20180828' '20180829'
do
  # 统一修改确认后标注的名字 --> 统一变成相同的大写日期
  data_path2="$data_dir/$date_type2/$date"
  echo $data_path2

  python3 "$tool_dir/ocr/crcb/cdata_rename.py" $data_path2

  # 生成临时的rename_mark及对应的label list
  # python3 "$tool_dir/ocr/crcb/crcb_copy.py" $data_dir $date_type2 $date
done

# # ssd批量跑rename_mark
# # python "$tool_dir/ocr/crcb/get_cd_roi_ssd.py" --labelmap_file $labelmap_file --mean_value $mean_value --model_def $model_def --model_weights $model_weights --image_resize $image_resize
# # 如果修改了get_cd_roi_ssd.py的参数就直接运行下面一行代码
# python "$tool_dir/ocr/crcb/get_cd_roi_ssd.py"

# # 对ssd定位的结果进行改名 修改下面代码中的路径
# python "$tool_dir/ocr/crcb/ca_re_ssd_roi_name.py"


