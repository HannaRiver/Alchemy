# 训练大写金额手写体/打印体操作手册

## 1. 处理训练数据

### 情况1. 标注为定位数据

### 1.1 扣取ROI并且用模型检查然后数据部门标注核实
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/get_roi4json.py --key_date 20181024 --rec_item CA
```
扣取的roi保存在/data1/ocr/data/org_print/checkdata/CA/return文件夹中

### 1.2 处理retun文件夹，用识别模型进行清洗
```
python /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/check_labeled_crnn.py --key_date 20181024 --rec_item CA
```
生成的数据会保存在/data1/ocr/data/org_print/checkdata/CA/check_right或者check_error中

### 1.3 对于check_right 生成网络所需要的数据

#### 1.3.1 生成训练所需要的数据(resize到rec文件夹中, 包含随机绕动位置拓增)
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/aug_resize2train.py --item_type CA --date 20181024 --check_state check_right
```

#### 1.3.2 生成增加训练数据的imglist
p.s. 这个路径自己把控，下面只是一个示例
```
ls /data1/ocr/data/rec/CA/train/20181024/check_right/* >> /data1/ocr/net/CRNN/CA/V0.0.1/data/20181024_check_right.txt
ls /data1/ocr/data/rec/CA/train/20181129/check_right/* >> /data1/ocr/net/CRNN/CA/V0.0.1/data/20181129_check_right.txt
```
跳转到3.2 给样本打标签
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/create_rec_label.py --img_txt /data1/ocr/net/CRNN/CA/V0.0.1/data/20190713add.txt --labeled_path /data1/ocr/net/CRNN/CA/V0.0.1/data/20190713add_labeled.txt --label_indx 0
```

#### 1.3.3 生成增加测试数据的imglist
p.s. 这个路径自己把控，下面只是一个示例
```
ls /data1/ocr/data/rec/CA/test/2018*/* >> /data1/ocr/net/CRNN/CA/V0.0.1/data/20190713addTest.txt
```
跳转到3.2 给样本打标签
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/create_rec_label.py --img_txt /data1/ocr/net/CRNN/CA/V0.0.1/data/20190713addTest.txt --labeled_path /data1/ocr/net/CRNN/CA/V0.0.1/data/20190713addTest_labeled.txt --label_indx 0
```


### 1.4 对于check_error 生成网络所需要的数据

#### 1.4.1 核实标签
核实check_error目录中的数据标签

#### 1.4.2 生成训练所需要的数据(保存在rec/CA/train/20181024/check_error/)
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/aug_resize2train.py --item_type CA --date 20181024 --check_state check_error

#### 1.4.3 生成增加训练数据的imglist
p.s. 这个路径自己把控，下面只是一个示例
```
ls /data1/ocr/data/rec/CA/train/20181024/check_error/* >> /data1/ocr/net/CRNN/CA/V0.0.2/data/20181024_check_error.txt
ls /data1/ocr/data/rec/CA/train/20181129/check_error/* >> /data1/ocr/net/CRNN/CA/V0.0.2/data/20181129_check_error.txt
```
跳转到3.2 给样本打标签
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/ca_create_rec_label.py --img_txt /data1/ocr/net/CRNN/CA/V0.0.2/data/20181129_check_error.txt --labeled_path /data1/ocr/net/CRNN/CA/V0.0.2/data/20181129_check_error_labeled.txt --label_indx 0
```

#### 1.4.4 生成增加测试数据的imglist
p.s. 这个路径自己把控，下面只是一个示例
```
ls /data1/ocr/data/rec/CA/test/2018*/* >> /data1/ocr/net/CRNN/CA/V0.0.1/data/20190713addTest.txt
```
跳转到3.2 给样本打标签
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/create_rec_label.py --img_txt /data1/ocr/net/CRNN/CA/V0.0.2/data/20181129_check_error.txt --labeled_path /data1/ocr/net/CRNN/CA/V0.0.2/data/20181129_check_error_labeled.txt --label_indx 0 --rec_item CA
```

## 情况2. 原始数据/未标注

### 2.1 获取日期批次数据
从数据分发者(小谈//192.168.20.252)下载小谈邮件通知处理好的日期数据， 运行下行脚本(20180814指日期批次号)
```
bash /data1/ocr/ocr_project/alchemy/ocr/crcb/data/download_shred_data.sh 20180814
```
#### 2.1.1 第一次使用
1. 修改alchemy/ocr/crcb/data/download_shred_data.py 中的默认参数，主要修改is_hd，False表示打印体，True表示手写体
2. 修改alchemy/ocr/crcb/data/download_shred_data.sh 中`classify_dir`和`data_dir`


###2.1 先进行样本识别+定位+打标签
修改/data1/ocr/ocr_project/alchemy/ocr/crcb/det2rec/run_det2rec4labeldata.py中的参数并运行下面代码
```
python /data1/ocr/ocr_project/alchemy/ocr/crcb/det2rec/run_det2rec4labeldata.py
```

### 2.2 将model_error的文件夹上传至数据部标注
上传地址: smb://192.168.30.41/ocr_up/OCR/HandWriting/CRCB/crcb_shred/print/CA

核实图片内容是否和第一位标签一致

### 2.3 将model_right文件个人核实一遍加入训练
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/aug_resize2train.py --item_type CA --date 20181024 --check_state model_right
```
### 2.4  将数据部返回结果放入return文件夹中
跳转至1.2流程然后继续1.2之后的流程

## 3. 标签生成模块

### 3.1.1 生成测试样本list
```
# 在终端中运行下面这行代码
ls /data1/ocr/data/rec/CA/test/CA_V010_test/* >> /data1/ocr/net/CRNN/CA/V0.0.1/data/test_img.txt
```

### 3.1.2 生成训练样本list
```
# 在终端中运行下面这行代码
ls /data1/ocr/data/rec/CA/train/CA_V012_train_labeld/* >> /data1/ocr/net/CRNN/CA/V0.0.1/data/train_img.txt
```

### 3.2 给样本打标签

#### Test集
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/create_rec_label.py /data1/ocr/net/CRNN/CA/V0.0.1/data/test_img.txt /data1/ocr/net/CRNN/CA/V0.0.1/data/test_labeled.txt -1
```

#### Train集
```
python3 /data1/ocr/ocr_project/alchemy/ocr/crcb/rec/create_rec_label.py /data1/ocr/net/CRNN/CA/V0.0.1/data/train_img.txt /data1/ocr/net/CRNN/CA/V0.0.1/data/train_labeled.txt -1
```

### 训练细节 

#### 1. 计算多少次迭代是一个epoch

修改solver中的`test_interval`为train_size/batch_size = 1000

修改slover中的`test_iter`为test_size/batch_size = 7755/4 = 1938.75

#### 2. 修改model/train.prototxt

修改及核实里面训练及测试的文件路径及参数

#### 3. 看准确度图
```
bash /data1/ocr/ocr_project/alchemy/caffe/alchemy/auto_draw_log.sh /data1/ocr/net/CRNN/CA/V0.0.1/logs/ca_v0.0.1_201907112232.log ctc 0 /data1/ocr/net/CRNN/CA/V0.0.1/logs/ca_v0.0.1_201907112232.png
```
