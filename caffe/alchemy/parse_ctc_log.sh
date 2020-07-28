#!/bin/bash
# Usage parse_log.sh caffe.log
# It creates the following two text files, each containing a table:
#     caffe.log.test (columns: '#Iters Seconds TestAccuracy TestLoss')
#     caffe.log.train (columns: '#Iters Seconds TrainingLoss LearningRate')

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
# echo $DIR
# echo "I am run...$(data-%Y%m%d%H%M")"
if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your.log"
exit
fi

LOG=`basename $1`
# 只提取Iteration需要解析的内容
sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > aux.txt
# 删除Waiting for data
sed -i '/Waiting for data/d' aux.txt
sed -i '/Optimization Done/d' aux.txt
sed -i '/prefetch queue empty/d' aux.txt
sed -i '/Iteration .* loss/d' aux.txt
sed -i '/Iteration .* lr/d' aux.txt
sed -i '/Train net/d' aux.txt
sed -i '/Snapshotting .*/d' aux.txt

grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep 'Test net output #0' aux.txt | awk '{print $11}' > aux1.txt
grep 'Test net output #1' aux.txt | awk '{print $11}' > aux2.txt

grep '] Solving ' $1 > aux3.txt
grep 'Testing net' $1 >> aux3.txt
$DIR/extract_seconds.py aux3.txt aux4.txt

echo '#Iters Seconds TestAccuracy TestLoss'> $LOG.test
paste aux0.txt aux4.txt aux1.txt aux2.txt | column -t >> $LOG.test
rm aux.txt aux0.txt aux1.txt aux2.txt aux3.txt aux4.txt

grep '] Solving ' $1 > aux.txt
grep ', loss = ' $1 >> aux.txt
$DIR/extract_seconds.py aux.txt aux4.txt
rm aux.txt

sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > aux.txt
# 删除Waiting for data
sed -i '/Waiting for data/d' aux.txt
sed -i '/Optimization Done/d' aux.txt
sed -i '/Iteration .* loss/d' aux.txt
sed -i '/Iteration .* lr/d' aux.txt
sed -i '/Test net/d' aux.txt
sed -i '/Snapshotting .*/d' aux.txt

grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ', lr = ' $1 | awk '{print $9}' > aux2.txt

grep 'Train net output #0' aux.txt | awk '{print $11}' > aux3.txt
grep 'Train net output #1' aux.txt | awk '{print $11}' > aux1.txt

if [[ "`cat aux0.txt |wc -l`" != "`cat aux1.txt |wc -l`" ]];then
  sed -i '$d' aux0.txt
  sed -i '$d' aux4.txt
fi

echo '#Iters Seconds TrainingLoss LearningRate TrainingAcc'> $LOG.train
paste aux0.txt aux4.txt aux1.txt aux2.txt aux3.txt | column -t >> $LOG.train
rm aux.txt aux0.txt aux1.txt aux2.txt aux3.txt aux4.txt