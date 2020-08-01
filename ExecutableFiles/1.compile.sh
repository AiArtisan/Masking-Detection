###
# @Author: xiaobo
 # @Email: 729527658@qq.com
 # @Date: 2020-06-06
 # @Description: convert the elf model to the shared library 
 # @Dependence: tensorflow 1.13, Vitis-AI Release 1.1
 ###
#!/bin/bash

echo "#############################################"
echo "Convert elf model to the shared library begin"
rm *.so
sudo gcc -fPIC -shared dpu_dpuCarModel_0.elf -o libdpumodeldpuCarModel.so
echo "Convert completed!"
echo "#############################################"
