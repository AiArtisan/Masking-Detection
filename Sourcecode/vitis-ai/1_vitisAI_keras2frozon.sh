###
# @Author: xiaobo
 # @Email: 729527658@qq.com
 # @Date: 2020-05-03
 # @Description: convert the keras model to a frozon model 
 # @Dependence: tensorflow 1.13, Vitis-AI Release 1.1
 ###
#!/bin/bash

# remove existing files
rm -rf ./frozon_result
mkdir frozon_result


# main process
echo "############################################"
echo "Convert keras h5 model to frozon model begin"
echo "Vitis AI 1.1"
echo "############################################"
python keras_to_tensorflow.py --input_model="./model.h5" --output_model="./frozon_result/model.pb"
echo "############################################"
echo "Convert completed"
echo "############################################"
