#!/bin/bash

classify_dir='/work/hena/scripts/caffe/classification'
# date='20180918'
date=$1
data_dir='/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item'
# data_dir='/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/20180925'


{
  for item in '大写金额' 支票大写金额 进账单大写金额
  do
      python $classify_dir/classify_class.py --item $item --date $date
      # rm -fr "$data_dir/$item/$date"
      # mv "$data_dir/$item/handwriting" "$data_dir/$item/$date"
      echo "$date $item done!"
  done
} &


{
  for item in 进账单日期 凭证日期 支票签发日期
  do
    python $classify_dir/classify_class.py --item $item --date $date
    # rm -fr "$data_dir/$item/$date"
    # mv "$data_dir/$item/handwriting" "$data_dir/$item/$date"
    echo "$date $item done!"
  done
} &


{
    for item in 小写金额 支票小写金额 进账单小写金额
    do
      python $classify_dir/classify_class.py --item $item --date $date
      # rm -fr "$data_dir/$item/$date"
      # mv "$data_dir/$item/handwriting" "$data_dir/$item/$date"
      echo "$date $item done!"
    done
} &


{
    for item in 付款人账号 收款人账号 支付密码
    do
      python $classify_dir/classify_class.py --item $item --date $date
      # rm -fr "$data_dir/$item/$date"
      # mv "$data_dir/$item/handwriting" "$data_dir/$item/$date"
      echo "$date $item done!"
    done
} &


wait

echo "=================== Classify shrd data Done! ==================="