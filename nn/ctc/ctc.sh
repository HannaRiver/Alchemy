#!/bin/bash

img_file_list="/work/hena/ocr/data/FinancialStatements/recognize/chn_wrong/titlewrong.txt"
lstm="/work/hena/scripts/caffe/classification/lstm.py"
ctc_lm="/work/hena/scripts/nn/ctc.py"
i=0

for line in $(cat $img_file_list)
do
    let i=i+1
    echo "[${i}]${line}"
    img_path=${line}
    python ${lstm} --img_path=${img_path}

    python3 ${ctc_lm}   
done