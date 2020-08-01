###
 # @Author: xiaobo
 # @Email: 729527658@qq.com
 # @Date: 2020-04-22 
 # @Description: Compile the quantized tf model to dpu model
 # @Dependence: tensorflow 1.13, Vitis-AI Release 1.1
 ###
#!/bin/bash

# delete previous results
rm -rf ./compile_result

#TARGET=ZCU104
NET_NAME=testModel
# DEPLOY_MODEL_PATH=vai_q_output

#ARCH=${CONDA_PREFIX}/arch/dpuv2/${TARGET}/${TARGET}.json
ARCH="u96pynq.json"

# Compile
echo "#####################################"
echo "COMPILE WITH DNNC begin"
echo "#####################################"
vai_c_tensorflow \
       --frozen_pb=./quantize_results/deploy_model.pb \
       --arch ${ARCH} \
       --output_dir=compile_result \
       --net_name=dpuCarModel \
       -e "{'save_kernel':'', 'mode':'Normal', 'dump':'Graph'}"
       #--mode debug\
      # --dump Graph\

echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"
