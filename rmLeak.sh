fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
# replcae nvidia* to nvidiai if want to free GPU_i
