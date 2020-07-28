# Alchemy


## 目录结构

以下表格是本修仙秘籍中所有目录的解释:

| 目录 | 解释
| ---- | -------------
| requirements.txt | 存放软件依赖的外部Python包列表
| caffe | caffe相关的一些操作
| imgaug | 数据增广的一些操作
| nn | 网络相关的若干优化及分析脚本(目前支持: ctc, ctd, 常用操作(lstm数据标签预处理等))
| ocr | OCR项目中特定的脚本工具(根据具体项目再做细分)
| test | 测试用例
| utils | 通用操作

## 开发规范

1. 请大家尽量遵守pep8(这里推荐一个中文版的[link](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/))进行开发.

## 开发流程

- fork 一份代码到自己的仓库
- 本地新建代码分支开发新功能、修 bug 等
- 开发完成后提交 merge request，有其他成员进行 code review。注意 commit 信息不能太随意，一般我们有一些提交模板。说明新增了哪些功能（需求文档链接）？修复了什么 bug（jira 链接附上并说明如何避免类似问题）？
- 确认代码没问题合并到主干，每次构建都会执行单测、pylint 检测等

## 开发环境