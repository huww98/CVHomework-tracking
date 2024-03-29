# 计算机视觉作业三-运动跟踪

## 用法

安装依赖
```shell
pip install -r requirments.txt
```

将数据放入`data`文件夹中，格式如下：

```
data
├── OTB
│   ├── Liquor
│   │   ├── groundtruth_rect.txt
│   │   ├── GOTURNresult.txt
│   │   ├── KCFresult.txt
│   │   ├── TLDresult.txt
│   │   └── ......
│   ├── Panda
│   │   ├── groundtruth_rect.txt
│   │   ├── GOTURNresult.txt
│   │   ├── KCFresult.txt
│   │   ├── TLDresult.txt
│   │   └── ......
│   └── ......
├── VOT
│   └── ......
└── .....
```

使用数据集名，Sequence名等建立目录层级结构，支持任意多级目录。每个Sequence的数据中需要包括：
* groundtruth_rect.txt 人工标注的跟踪边界框
* [ALGO]result.txt 以每个算法的名字开头的文件，记录该算法输出的边界框

每个txt文件包含的行数需等于该Sequence的帧数，每一行包含`x,y,w,h`的数据，或`Tracking failure detected`表示算法跟踪失败。

运行以下命令生成所有图表
```
python graph.py
```

或运行以下命令生成特定Sequence的图表
```
python graph.py OTB/Liquor OTB/Panda ...
```

生成的图表将位于`output/graph`目录下，将会保留输入数据的层级结构，并在每一级子目录绘制汇总图表。最外层将会有额为的`all.png`汇总所有输入数据。
