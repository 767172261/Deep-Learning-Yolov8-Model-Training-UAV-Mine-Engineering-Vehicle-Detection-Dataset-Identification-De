深度学习Yolov8模型训练无人机矿场工程车检测数据集 识别检测无人机挖掘机检测数据集 无人机推土机数据集检测 无人机起重机数据集检测等10类
@[toc]
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e3c786e0dfe44b2a596e475f3928324.png)
无人机矿场工程车检测数据集 4900张 工程车voc yolo
​​![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/53cf7900053141eab06d8450fbe66489.png)



### 🚜 工程车辆目标检测数据集统计表

| 序号 | 分类名称           | 英文类别名               | 图片张数 | 标注个数 |
|------|--------------------|--------------------------|----------|-----------|
| 1    | 挖掘机             | excavator                | 1,086    | 1,356     |
| 2    | 自卸卡车           | dump_truck               | 1,074    | 1,816     |
| 3    | 压路机             | compactor                | 538      | 583       |
| 4    | 移动式起重机       | mobile_crane             | 529      | 589       |
| 5    | 塔式起重机         | tower_crane              | 169      | 235       |
| 6    | 轮式装载机         | wheel_loader             | 851      | 922       |
| 7    | 混凝土搅拌车       | concrete_mixer_truck     | 376      | 427       |
| 8    | 反铲装载机         | backhoe_loader           | 666      | 693       |
| 9    | 推土机             | dozer                    | 492      | 536       |
| 10   | 平地机             | grader                   | 622      | 641       |
| **总计** | -                  | -                        | **4,981**| **7,798** |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/79a08e9800e34ab4a527a2bac4d79955.png)
矿场工程车检测数据集， **4981张图像**、**7798个标注目标**，共 **10类工程车辆**。构建深度学习目标检测系统（YOLOv8）的完整流程，包括：

---

## ✅ 一、数据格式说明

数据是 **VOC XML 格式** 和 **YOLO TXT 格式** 的标注文件。

- **VOC XML**：每个图片对应一个 `.xml` 文件，包含边界框和类别信息。
- **YOLO TXT**：每个图片对应一个 `.txt` 文件，每行表示一个目标：`class_id x_center y_center width height`（归一化坐标）。

---

## ✅ 二、目录结构建议

```
mining_vehicle_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

---

## ✅ 三、data.yaml 配置文件

```yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 10  # 类别数
names: ['excavator', 'dump_truck', 'compactor', 'mobile_crane', 'tower_crane',
        'wheel_loader', 'concrete_mixer_truck', 'backhoe_loader', 'dozer', 'grader']
```

---

## ✅ 四、模型训练（以 YOLOv8 为例）

### 1. 安装依赖

```bash
pip install ultralytics
```

### 2. 开始训练

```bash
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

- `model`: 可选 yolov8n/s/m/l/x
- `epochs`: 训练轮次，建议从100开始
- `imgsz`: 图像尺寸，根据硬件选择（如640或1280）
- `batch`: 批量大小，根据显存调整

---

## ✅ 五、推理与部署

### 1. 单图推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/train/weights/best.pt')

# 推理
results = model('test.jpg')

# 显示结果
for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
    im.show()
```

### 2. 视频推理

```python
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Mining Vehicle Detection", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
```

### 3. 批量推理（文件夹）

```bash
yolo task=detect mode=predict model=best.pt source="path/to/images" save=True
```

---

## ✅ 六、性能评估

```bash
yolo task=detect mode=val model=best.pt data=data.yaml
```

输出指标：
- mAP@0.5
- mAP@0.5:0.95
- Precision / Recall
- FPS（帧率）

---

## ✅ 七、可视化工具推荐

- **LabelImg / LabelYOLO**：用于查看标签
- **TensorBoard**：监控训练过程
- **Roboflow Universe**：可上传数据进行增强、格式转换等操作

---

## ✅ 八、无人机集成建议（矿场巡检）

如果你计划将此模型部署到 **无人机巡检系统中**，可以考虑以下方案：

### 硬件平台：
- NVIDIA Jetson Nano / Xavier NX / Orin Nano
- DJI M30/M30T 无人机 + RTSP 流传输

### 软件架构：
- 实时视频流接收（OpenCV / GStreamer）
- 模型推理（TensorRT优化YOLO）
- 目标识别 + 报警机制（MQTT/WebSocket推送）

---

## ✅ 九、代码打包建议

使用如下命令将你的项目打包成 Python 包或 Docker 镜像：

### 使用 PyInstaller 打包为可执行程序

```bash
pip install pyinstaller
pyinstaller --onefile detect_app.py
```

### 使用 Docker 构建镜像

```Dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "detect_app.py"]
```

---

## ✅ 十、附录：类别分布统计（供参考）

| 类别名              | 图片数量 | 标注数量 |
|---------------------|----------|-----------|
| excavator           | 1086     | 1356      |
| dump_truck          | 1074     | 1816      |
| compactor           | 538      | 583       |
| mobile_crane        | 529      | 589       |
| tower_crane         | 169      | 235       |
| wheel_loader        | 851      | 922       |
| concrete_mixer_truck| 376      | 427       |
| backhoe_loader      | 666      | 693       |
| dozer               | 492      | 536       |
| grader              | 622      | 641       |
| **总计**            | **4981** | **7798**  |

---

以上文字及代码仅供参考。
