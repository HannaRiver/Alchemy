import os


fw = open('/work/ocr/card/driving_license/model/V0.2.0/data/20190310_test.txt', 'w')
img_root = '/work/ocr/card/driving_license/data/fake/jiashizheng01'
for item in ['jiashizheng04']:
    img_dir = os.path.join(img_root, item, 'train_data')
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        fw.write(img_path + '\n')
fw.close()