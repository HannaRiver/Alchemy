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