
# An-assistant-to-thr-basic-principles-of-Marxism

## 环境的配置
使用pip安装requirements中所需要的包（conda可能出现不兼容的问题），同时pyhon版本建议在3.10以上
## 知识库的构建
- 先将所需训练的原始材料放入mayuan_raw_data文件夹
- 运行genetrate_database，会自动将知识库构建在database_agent_mayuan这个文件夹中，后续更新知识库的时候再次运行即可
## 模型调用
运行mayuan_agent即可调用模型，同样模型的整体架构搭建也在这个脚本中，可通过修改架构实现不同的功能，在终端中输入quit即可退出模型
