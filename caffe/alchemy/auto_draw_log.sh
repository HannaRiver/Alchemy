log_path=$1
caffe_type=$2
draw_type=$3
save_path=$4
tool_dir=$(cd "$(dirname "$0")";pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
tool_path=$tool_dir/plot_training_log.py
check_time=5s

if [[ $# != 4 ]];then
  echo "========== Usage: =========="
  echo "bash ./auto_draw_log.sh /path/to/draw.log caffe_type[ssd/ctc] draw_type[0-7/0-9] /where/to/save.png"
  echo    "Supported draw types(ssd version)":
    
  echo  0: Test accuracy  vs. Iters
  echo  1: Test accuracy  vs. Seconds
  echo  2: Test loss  vs. Iters
  echo  3: Test loss  vs. Seconds
  echo  4: Train learning rate  vs. Iters
  echo  5: Train learning rate  vs. Seconds
  echo  6: Train loss  vs. Iters
  echo  7: Train loss  vs. Seconds

  echo    "Supported draw types(ctc version)":

  echo  0: Test accuracy  vs. Iters
  echo  1: Test accuracy  vs. Seconds
  echo  2: Test loss  vs. Iters
  echo  3: Test loss  vs. Seconds
  echo  4: Train accuracy  vs. Iters
  echo  5: Train accuracy  vs. Seconds
  echo  6: Train learning rate  vs. Iters
  echo  7: Train learning rate  vs. Seconds
  echo  8: Train loss  vs. Iters
  echo  9: Train loss  vs. Seconds

  echo "怎么都运行不了？ --> Email: hena@em-data.com.cn"

  exit
fi


i=1
sys_language="大小："
log_star_stat="`stat ${log_path}|grep ${sys_language}`"

if [[ -z "${log_star_stat}" ]];then
  sys_language="Size: "
fi

while [[ true ]]; do
  log_old_stat="`stat ${log_path}|grep ${sys_language}`"
  sleep ${check_time}
  ((i=i+1));
  log_new_stat="`stat ${log_path}|grep ${sys_language}`"
  
  if [[ `echo ${log_old_stat}` != `echo ${log_new_stat}` ]];then
    python3 ${tool_path} ${draw_type} ${save_path} ${caffe_type} ${log_path}
  fi

  flag=$[$i % 120]
  if [[ $flag -eq 0 ]];then
    log_check_stat="`stat ${log_path}|grep ${sys_language}`"
    if [[ `echo ${log_star_stat}` == `echo ${log_check_stat}` ]];then
      break
    fi
  fi
done

echo "========== 模型要么崩了要么训完了，去看看吧！=========="
echo "Img save dir --> $save_path"
echo "没写路径在这里 --> $DIR/$save_path"