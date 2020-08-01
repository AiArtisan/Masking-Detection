#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The DpuCar is a module which contains the DpuCar class and the related common function

By xiaobo
Contact linxiaobo110@gmail.com
Created on  June 7 22:10 2020
"""

# Copyright (C)
#
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

import PIL
import IPython
from io import BytesIO as StringIO
from IPython.display import display
from IPython.display import clear_output
import cv2
from dnndk import n2cube
import numpy as np
from numpy import float32
import os
import matplotlib.pyplot as plt
import time
class DpuCar(object):
    def __init__(self, dpu_task, dpu_input_node="x_input_Conv2D", dpu_output_node="y_out_MatMul", dpu_img_size=128):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 160)
        self.cap.set(4, 120)
        print(self.cap.get(3), self.cap.get(4))
        print(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.dpuInputNode = dpu_input_node
        self.dpuOutputNode = dpu_output_node
        self.dpuTask = dpu_task
        self.dpuImgSize = dpu_img_size
        
    def get_image(self, idx=0):
        """
        get a image from sensor, donot care which kind of cam is used.
        Args:
            idx: the index of sensor, default is 0
        """
        if idx == 0:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("Please connect the camera!")
                return False
        else:
            print("The index should be 0!")
            
    def dpuPredictSoftmax(self, img_input):
        img_scale = cv2.resize(img_input,(self.dpuImgSize, self.dpuImgSize), interpolation = cv2.INTER_CUBIC)
        img1_scale = np.array(img_scale, dtype='float32')
        if np.max(img1_scale) > 1:
            img1_scale = img1_scale / 255.
        input_len = img1_scale.shape[0] * img1_scale.shape[1] * img1_scale.shape[2]
        n2cube.dpuSetInputTensorInHWCFP32(self.dpuTask, self.dpuInputNode, img1_scale, input_len)
        n2cube.dpuRunTask(self.dpuTask)
        conf = n2cube.dpuGetOutputTensorAddress(self.dpuTask, self.dpuOutputNode)
        channel = n2cube.dpuGetOutputTensorChannel(self.dpuTask, self.dpuOutputNode)
        outScale = n2cube.dpuGetOutputTensorScale(self.dpuTask, self.dpuOutputNode)
        size = n2cube.dpuGetOutputTensorSize(self.dpuTask, self.dpuOutputNode)
        softmax = n2cube.dpuRunSoftmax(conf, channel, size//channel, outScale)
        pdt= np.argmax(softmax, axis=0)
        return pdt

class CommonFunction(object):   
    @classmethod
    def img2display(cls, img_mat):
        ret, png = cv2.imencode('.png', img_mat)
        encoded = base64.b64encode(png)
        return Image(data=encoded.decode('ascii'))

    @classmethod
    def show_img_jupyter(cls, img_mat):
        img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)
        f = StringIO()
        PIL.Image.fromarray(img_mat).save(f, 'png')
        IPython.display.display(IPython.display.Image(data=f.getvalue())) 
       
    @classmethod
    def clear_output(cls):
        clear_output(wait=True)