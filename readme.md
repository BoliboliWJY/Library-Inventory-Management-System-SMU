# 当前计划

现在要做的是完成CNN基本逻辑模块，后续复杂的内容再进行添加。

## 一维码入库

使用一维码信息作为第一方案（考虑使用csv/sql作为数据库，大概率会用sql）

## 分类模型（当一维码损坏/不可用时）

但是似乎仅CNN是不行的，因为不可能为每本图书拍下几千张图片，平均每本1-2张最多了，因此需要考虑CNN（预训练）+Transfer Learning。

确保能够正确完成分类任务，还要有OCR功能辅助识别，**这个后面再说**

## 简介

这是一个用于SMU爱心书屋的图书进销存系统，正处于开发阶段
