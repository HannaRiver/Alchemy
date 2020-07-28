# 处理违章手持车牌数据流程

## 1、 数据标注准备

1. 进入'/work/hena/ocr/pro/CLPR/generate_dir'
2. 从txt.txt中剪切1k张数据保存为batchXk.txt(X为第几个批次)
3. 获取数据
    ```
    python3 /work/hena/scripts/ocr/clpr/data/get_car_data.py 5
    ```
4. 将result_batch5、 txt_batch5、image_batch5 放入batch5中
5. 让吴昊跑最好的定位模型定位roi_batch5 返回pics文件夹
6. 上传pics文件夹至det/suzhou_20180531/batch5_上传日期
7. 将数据部返回的定位结果 交给张祥祥进行数据打标签，并上传至rec文件夹中

## 2、 数据部门标注好定位数据后整合数据给识别模块
1. 对于数据部门标注的结果 --> 修改下面代码的路径并运行
    ```
    cd ./alchemy
    python3 ./utils/json2roi.py
    ```
2. 将数据部门标注的mark1放入batchK中，并重名名为det_v002_error
 
3. 对于det模型定位准确的结果，获取正确的list 并根据txt抠图保存
    ```
    cd ./alchemy/ocr/data
    python3 ./get_det_right_list.py 5
    ```
4. 对于数据部返回的标注数据和检测的含txt文件数据，利用gen_rectdata.py脚本生成xml文件

## 2、 数据部门标注好识别结果后

1. 将识别结果放入'/work/hena/ocr/pro/CLPR/generate_dir/batchXk'中（文件夹名字为batchXk_年月日）
2. 整合所有数据成一个原图与json的形式(如果数据批次改了，再修改info2unifiedJson.py中的参数)
    保存地址默认为：/work/hena/ocr/data/CLPR/shuzhou_20180531/batchXk
    ```
    cd /work/hena/scripts/ocr/clpr/data
    python3 info2unifiedJson.py 2 20181211
    ```
3. 让吴昊和张祥祥获取数据
