# 解析驾驶证数据流程

## 假数据
1. 将假数据分批次处理，每次处理1000张
2. 新建batchk文件夹，将batchk.txt放入batchk文件夹中
2. 修改``` ./idcard/data/json2owner.py ``` 
3. 第一次训模型的时候 统计数据的meanvalue