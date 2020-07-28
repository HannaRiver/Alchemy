Step1: 生成测试，训练的图片和标签列表 --> test_img.txt, test_label.txt, trainval_img.txt, trainval_label.txt
Setp2: 将labelmap放在上面所在的文件夹(data_list_dir)下
Step3: 修改train/test/solver.prototxt
Step4: ./ssd_tranin.sh