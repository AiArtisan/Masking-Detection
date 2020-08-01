### 
# @Author: xiaobo
 # @Email: 729527658@qq.com
 # @Date: 2020-04-20 
 # @Description: quantize frozon tf model 
 # @Dependence: tensorflow 1.13, Vitis-AI Release 1.1
 ###
#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment
#conda activate decent_q3

# generate calibraion images and list file
#python generate_images.py

# remove existing files


# run quantization
echo "#####################################"
echo "Quantize begin"
echo "Vitis AI 1.1"
echo "#####################################"
vai_q_tensorflow quantize \
  --input_frozen_graph ./frozon_result/model.pb \
  --input_nodes x_input \
  --input_shapes ?,160,160,3 \
  --output_nodes y_out/Softmax \
  --method 1 \
  --input_fn graph_input_fn.calib_input \
  --gpu 0 \
  --calib_iter 50 \

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"

