# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5-v1.0
File Name: main.py
Author: chenming
Create Date: 2021/11/9
Description：
-------------------------------------------------
"""
# 首先检查文件，把存在的目录删除
import os
os.chdir("/project/train/src_repo/")
# 启动数据处理
os.system("python server_data_gen.py")
os.system("python train_server.py >> /project/train/log/log.txt")
# 启动训练
