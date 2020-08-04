# AI-Masking-Detection
Object Detection About Masking For Xilinx Summer School
[中文版](README-zh.md) | English version

## Introduction
​		In response to the need for mask wearing recognition in the prevention and control of the epidemic (COVID-19), based on Xilinx's latest Vitis-AI tool, combined with a self-designed image recognition network, an AI mask wearing recognition system was quickly developed. The final recognition rate can reach more than 88%, and it can distinguish the conditions of wearing a mask correctly, without a mask, wearing a mask by mistake, covering the mouth, wearing a scarf, etc.

## Must have to run

1. Ultra96 V2 board, SD card
2. Network cable, power cable, microUSB data cable
3. U96-pynq2.5 image, upgrade to support DPU function
4. Driver-free USB camera

## Run steps

1. Clone the github repository to the jupyter_notebook directory of Ultra96.
2. On the Ultra96 terminal, after cd enters the repository folder, perform initialization operations:
``` bash
sudo python3 ./setup.py
```
> Requires administrator authority to change file attributes during initialization
3. Connect the USB camera, open the browser, enter the IP address, you can enter the jupyter Notebook.
4. On the user's PC, follow the instructions of Jupyter Notebook to run the program step by step to see the effect.

## Experimental results

#### Use the verification dataset stored in the SD card for testing:

![](image\结果1.jpg)
![](image\结果2.jpg)

#### Use the USB camera to recognize the mask wearing in real time:
![](image\结果3.png)

## Feedback and communication

Welcome friends who love AI and FPGA design to contact me by email`my e-mail：993987093@qq.com`

