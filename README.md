# Alchemy

Alchemy是何娜小分队(还没想好超级无敌厉害的队名, 目前成员: 吴昊、张祥祥、王林华、何娜)日常工作开发中使用的脚本工具，其中包含caffe的模型训练，前向，日志分析; 各个项目私有工具(给自己用的就没考虑你们要不要); 通用小脚本。

## 目录结构

以下表格是本修仙秘籍中所有目录的解释:

| 目录 | 解释
| ---- | -------------
| caffe | caffe相关的一些操作
| imgaug | 数据增广的一些操作
| nn | 网络相关的若干优化及分析脚本(目前支持: ctc, ctd, 常用操作(lstm数据标签预处理等))
| ocr | OCR项目中特定的脚本工具(根据具体项目再做细分)
| test | 测试用例
| utils | 通用操作

### caffe

以下表格是caffe目录中子目录的解释:

| 目录 | 解释
| ----- | ---------------
| alchemy | 模型分析工具
| classification | 分类模型(norm, lstm)
| detection | 检测模型(ssd)
| train | 模型训练脚本(目前支持: norm, ssd; lmdb, finturn)

#### alchemy

以下表格是alchemy目录中子文件的解释:

| 文件 | 解释
| ----- | -------------
| auto_draw_log.sh | 自动画日志图的脚本(*主函数* 运行方法直接运行看帮助)
| extract_seconds.py | 解析log中时间的代码
| parse_ctc_log.sh | 解析ctc log的脚本
| parse_ssd_log.sh | 解析ssd log的脚本
| plot_training_log.py | 画训练图目前支持(ssd, lstm)

#### classification

以下表格是classification目录中子文件的解释:(使用时需改caffe路径)

| 文件 | 解释
| ----- | --------------
| classify_class.py | 普通的分类网络前向
| lstm.py | lstm网络前向

#### detection

以下表格是detection目录中子文件的解释:(使用时需改caffe路径)

| 文件 | 解释
| ------- | -------------
| ssd_detect.py | 定位网络前向(目前支持ssd, textboxes++)

#### train

以下表格是train目录中子文件的解释:(使用时需要修改bash路径)

| 文件 | 解释 
| -------- | ------------
| crcb_shred_re_train.sh | 常熟银行手写体二分类finturn训练脚本
| crcb_shred_train.sh | 常熟银行手写体二分类一键训练脚本
| lmdb_create_data.sh | 打包lmdb脚本
| ssd_re_train.sh | ssd网络finturn训练脚本
| ssd_train_readme.txt | 使用一键训练ssd网络的操作文档
| ssd_train.sh | ssd网络一键训练脚本(包括数据准备)
--------------------------------------

### nn

以下表格是nn目录中子目录的解释:

| 目录 | 解释
| ------ | --------------
| ctc | ctc的若干优化(目前支持: 带LM约束的ctc)
| ctd | ctd(Curve Text Detector)相关代码
| utils | 通用操作

#### ctc

以下表格是ctc目录中子文件的解释:

| 文件 | 解释
| ------- | -------------
| chn_tab.pkl | 打包的中文标签list(需不断更新)
| ctc.py | 带LM约束的ctc代码(运行效率有待优化)
| ctc.sh | 结合python2 lstm的结果做中文分析(python3)的脚本(主要目的是通过bash结合python2&3)
| fs_lm_info.pkl | 打包的LM信息
| FS_LM.py | 分析特定场景(FS --> Financical Statements)并获取LM信息的代码
| prob.npy | ctc.py保存的概率分布

#### ctd

以下表格是ctd目录中子文件的解释:

| 文件 | 解释
| ------- | -------------
| json2CtdLabel.py | 数据部标注的rect(json)转化为ctd训练的txt(需要修改路径)

#### utils

以下表格是utils目录中子文件的解释:

| 文件 | 解释
| ------- | -------------
| create_label.py | 生成lstm训练所需的label(目前支持:crcb/CA,CDate,其他项目可以参考单独写)
| get_mean_value.py | 获取一批图像的均值、均高、均宽、及宽高极值(需修改路径)
--------------------------------------

### ocr

以下表格是ocr目录中子目录的解释:

| 目录 | 解释
| ----- | ------------------
| bgnet | 表格线网络项目
| card | 证件类项目
| crcb | 常熟银行手写体项目
| fs | 金融报表相关项目
| ocr_base_log2report.py | 解析工程输出日志文件转成识别率的工具(目前支持: 驾驶证、行驶证、转账支票)

#### bgnet

以下表格是bgnet目录中子文件的解释:

| 文件 | 解释
| ------- | ----------------
| mk_resize_mask.py | 生成resize之后的bgnet的mask图(横线竖线, 需修改路径)

#### card

以下表格是card目录中子目录的解释:

| 目录/文件 | 解释
| -------- | --------------
| idcard | 处理身份证数据相关代码

##### idcard

以下表格是card/idcard目录中子目录的解释及文件说明:

| 目录/文件 | 解释
| -------- | --------------
| address | 处理身份证地址相关代码
| address/chars_recognise.py | 字符识别模块
| address/chars_segment.py | 地址栏字符分割
| data| 处理身份证数据相关代码
| data/download_idcard_json.py | 下载陈伟造假的身份证信息包含的json文件

#### clpr

以下表格是clpr目录中子目录的解释及文件说明:

| 目录/文件 | 解释
| --------| ----------------
| cls | 车辆相关分类模型
| cls/car_head_cls.py | 车头车尾二分类模型
| cls/plate_cls.py | 车牌类别分类(0:白 1:单黄 2:双黄 3:蓝 4:新能源)
| data | 处理clpr数据相关代码
| data/vai_data | 处理车检数据相关代码及操作文档
| data/vai_data/get_newenergy_plate.py | 根据csv对新能源的数据进行提取
| data/vai_data/pic_check_rec_data.py | 根据模型识别结果对车检数据打标签
| data/vai_data/sorting_checked_vaidata.py | 整理车检已确认/修改标签后的数据，将车牌(roi)对应的车辆图片及定位txt保存在一个batch中
| data/vai_data/vai_plate_rename.py | 根据csv对车检数据进行重命名
| data/vai_data/VaiDataReadMe.md | 车牌识别(CLPR)处理车检(VAI)数据流程及操作手册
| data/example.json | 数据统一存储备份的标准json文件
| data/gen_rectdata.py | 处理数据部标注的四边形数据和预标定产生的txt数据
| data/get_car_data.py | 根据违章提供的generate_dir提取若干批次的数据
| data/get_det_right_list.py | 获取模型定位准确的数据列表
| data/info2unifiedJson.py | 转换为统一标准json文件代码(使用见data/README.md)
| data/isPlateNum.py | 判断str是否满足车牌格式
| data/README.md | 处理违章数据的操作流程手册
| det | 车辆相关检测模型
| det/plate_locate_ssd.py | 车牌定位ssd模型
| det/plate_locate_textboxes.py | 车牌定位textboxes模型
| edc | Error Detection and Corrections
| edc/refuse2plate.py | 车牌拒识功能
| rec | 车牌识别模型
| rec/plate_recognize_lstm.py | 车牌识别lstm模型

#### crcb

以下表格是crcb目录中子目录的解释及文件说明:

| 目录/文件 | 解释
| --------| ----------------
| ca | 大写金额模块代码
| ca/ca_re_ssd_roi_name.py | 根据人工check结果对roi进行rename
| ca/ca_rename.py | 根据人工check的roi的label对原始的mark，xml，json进行改名(大写金额类)
| ca/ca2lca.py | 大写金额 --> 小写金额 
| ca/get_ca_datalist.py | 初步解析完银行数据之后获取大写金额训练及测试数据
| ca/get_ca_error_bnbox.py | 获取识别不正确的定位结果及原图
| ca/get_ca_roi_ssd.py | 获取大写金额定位结果并支持数据拓增(带旋转角度)
| ca/lca2ca.py | 批量将大写金额数字标签 --> 大写标签(小写金额 --> 大写金额)
| ca/mk_ca_aug_roi.py | 对大写金额识别数据进行数据拓增
| ca/mk_ca_resize_data.py | 对大写金额识别数据进行不变形resize
| cdata | 大写日期模块代码
| cdata/cd2lcd | 大写日期 --> 小写日期
| cdata/cdata_rename | 根据人工check的roi的label对原始的mark，xml，json进行改名(大写日期类)
| cdata/get_cd_datalist.py | 初步解析完银行数据之后获取大写日期训练及测试数据
| cdata/get_cd_roi_ssd.py | 获取大写日期定位结果并支持数据拓增(带旋转角度)
| cdata/get_cd_roi_ssd.sh | 批量扣取大写日期roi脚本
| cdata/lcd2cd.py | 小写日期 --> 大写日期
| cdata/mk_cd_aug_roi.py | 对大写日期识别数据进行数据拓增
| cdata/mk_cd_resize_data.py | 对大写日期识别数据进行不变形resize
| cls | 手写非手写分类
| cls/HandwritingClassify.py | 手写非手写二分类模型
| data | crcb数据相关处理文件夹
| data/download_shred_data.py | 从数据分发者(谈咏东)获取自己所需大类并按照日期分发
| rec | 训练及处理识别模型数据及网络模块
| rec/data/get_roi4json.py | 通过json获取roi图像，通常用于处理标注好的定位数据
| respy | 解析常熟银行数据模块(未确认)
| README.md | 处理crcb数据及模型的操作流程手册

#### fs

以下表格是fs目录中子文件的解释:

| 文件 | 解释
| ------ | --------------
| ocr_log_case.txt | 能够支持解析fs log的示例log
| ocr_log2info.py | 解析fs log的代码
| title_rect.py | 报表表头定位分类的数据拓增及回收代码(私用)
-------------------------------------------

### tests

以下表格是tests目录中子文件的解释:

| 文件 | 解释
| ------ | -------------
| test4utils.py | utils文件夹下面的函数测试代码
--------------------------------------------------

### utils

以下表格是utils目录中子文件的解释:

| 文件 | 解释
| ------ | -------------
| doc | 对于doc处理的相关代码
| doc/doc2docx4win.py | 只能在windows下运行，将doc转换成docx
| doc/docx2txt_my.py | docx转换成txt
| quad/get_quad_roi.py | 將四个点的坐标转换成旋转过后的roi
| batch_down8aria2.py | 通过aria2c对link批量下载示例
| char_trans.py | 数据转化和处理(目前支持: 删除字符串中的中文、全角转半角、小数标准化、去除标点、忽略括号(主要根据ocr/fs项目设计))
| combine_img.py | 拼接两张图片
| EditDistance.py | 编辑距离，用于分析模型混淆矩阵
| file2list.py | 读取文件加载为一个list(普通txt, vir plate csv)
| json2roi.py | 各种从json中提取roi的操作
| resize_img.py | 若干不同方式的resize(目前支持:不变形resize)
| txt2roi.py | txt保存的坐标转换为roi
| write_xml.py | xml相关的若干操作(写入、修改、获取特定信息)
| random_crop.py | 对图片进行padding后随机裁剪并修改相应xml文件
--------------------------------------------------
