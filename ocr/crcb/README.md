| file name | Exp 
| --------- | --- 
| sh | 脚本文件夹，一般是批量处理或者连接不同版本的python或多进程的功能
| ca | 处理大写金额类数据文件夹
| lca2ca.py | 金额标签转为大写金额标签
| lcd2cd.py | 小写日期转为大写日期标签
| cd2lcd.py | 大写日期标签转为小写日期
| get_cd_roi_ssd.sh | 批量处理获取日期ssd跑出的roi数据，并对数据改名


# 解析crcb数据流程(从银行直接获取的数据)
1. ``` python repicture.py``` #  获取文件映射关系--> pictures.txt
2. 将pictures.txt用*gb2312*打开并已*utf-8*保存(利用vscode转码)
3. 将pictures文件夹保存为碎片日期文件夹，并将picture.txt放入其中，只保存图片的.bin文件
    ```
    mkdir 20180918
    mv pictures.txt 20180917/pictures.txt
    cd 20180918
    mkdir pictures
    cp ../pictures/dbvis* pictures/
    rm -rf ../pictures
    ```
4. 将碎片数据重命名并且按照项目分开
    ```
    python3 pickpic.py '20180918'
    ```
5. 将已分好的数据分手写非手写
    ```
    bash classify_shrd.sh 20180917
    ```
6. 删除handwriting并且改名为日期
    ```
    bash classify_shrd_step2.sh 20180917
    ```

# 对中文碎片进行初步分类

    ```
    bash sh/classify_chn_shrd.sh
    ```

# 迭代训练数据流程

## 大写金额类
1. 确认手写非手写文件夹，并重命名或者运行下面代码
   ```
   bash classify_shrd_step2.sh 20181018
   ```
2. 生成ssd 扣出来的roi 需要修改代码中的日期
    ```
    python get_ca_roi_ssd.py
    ```
3. lstm识别（需要修改日期） 获取不能识别的roi对应的原图放入date+'check_result' 能识别的roi放入CA/CA/date/ssd_roi
    ```
    python get_ca_error_bnbox.py
    ```
4. 再次对check_result运行ssd 生成bnbox图 打开tmp_function
    ```
    python get_ca_roi_ssd.py
    ```

# 训练模型流程

## 大写金额识别模型

### 数据部分
1. 生成数据txt文档

2. 生成对应的label文档（需要修改里面的txt路径及保存label文档的路径）
    ```
    create_label.py
    ```

3. 合成训练用的txt
    ```
    paste -d' ' sub_num.txt num_label.txt >> 0508_num_train.txt
    ```

### 网络部分

1. solver 地址
    /work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/model/V0.0.3/solver.prototxt

2. 修改train.prototxt

### 训练
```
/home/hena/caffe-ocr/buildcmake/tools/caffe train -gpu 0 -solver /work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/v0.0.1/solver.prototxt -weights /work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/v0.0.1/weights/ca_iter_110000.caffemodel 2>&1 |tee /work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/logs/2018-09-17-20-48.logs
```

## 大写日期识别模型

### 数据部分
1. 进行数据拓增，并且保存拓增后的图
1. 获取需要resize的图片列表
    ```
    ./cdate/get_cd_datalist.py
    ```
1. 数据准备
    ```
    ./cdate/mk_cd_resize_data.py 

    ```