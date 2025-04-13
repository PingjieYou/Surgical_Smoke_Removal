# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2024/5/5 20:43
# @Email   : youpj2000@gmail.com
# @Function:

import csv
import shutil

line = 0
first_line = True

trainA = []
trainB = []

trainA_path = 'D:\Dataset\Cholec80_Desmoking\\trainA\\'
trainB_path = 'D:\Dataset\Cholec80_Desmoking\\trainB\\'

# 打开CSV文件
with open('./assets/pairs.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 读取CSV文件的每一行
    for row in csv_reader:
        if first_line:
            first_line = False
            continue

        image_path = row[0].split(';')[1]
        if line % 2 == 0:
            trainA.append(image_path)
        else:
            trainB.append(image_path)
        line += 1

for i in range(len(trainA)):
    shutil.copy(trainA[i], trainA_path + str(i) + '.bmp')
    shutil.copy(trainB[i], trainB_path + str(i) + '.bmp')
