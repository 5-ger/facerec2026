# 智能门禁系统

基于深度学习的人脸识别门禁系统，实现**活体检测 → 身份识别 → 门禁决策**三步骤流水线。

## 技术栈

- **深度学习框架**：PyTorch / MindSpore
- **模型架构**：FaceNet（4层CNN）+ ArcFace 角度边际损失
- **人脸检测**：OpenCV DNN SSD（回退 Haar Cascade）
- **活体检测**：自研方差眨眼检测（眼部区域像素强度方差时序分析）
- **GUI**：PyQt5 暗色主题
- **数据库**：SQLite

## 项目结构

```
facerec2026/
├── main.py                 # 程序入口
├── train.py                # 模型训练脚本
├── core/
│   ├── camera.py           # 摄像头采集线程
│   ├── face_detector.py    # 人脸检测 + 位置验证
│   ├── face_recognizer.py  # 人脸识别推理引擎
│   ├── liveness.py         # 眨眼活体检测
│   └── database.py         # SQLite 数据库操作
├── gui/
│   ├── main_window.py      # 主窗口 GUI（监控/注册/管理）
│   ├── register_dialog.py  # 注册对话框
│   └── styles.py           # 暗色主题样式
├── models/                 # 模型文件目录
└── raw/                    # 训练数据目录（按人名分文件夹）
```

## 快速开始

### 1. 准备模型文件

将 SSD 人脸检测模型放入 `models/` 目录：
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

### 2. 安装依赖

```bash
pip install opencv-python numpy Pillow PyQt5
```

MindSpore 或 PyTorch 二选一，按实际使用的框架安装。

### 3. 训练模型（可选，无预训练权重时）

```bash
# 在 raw/ 下按人名建文件夹放入照片，然后：
python train.py
```

### 4. 启动系统

```bash
python main.py
```

## 功能说明

| 模式 | 功能 |
|------|------|
| 监控模式 | 实时人脸检测 + 活体判定 + 身份识别 + 门禁日志 |
| 注册模式 | 摄像头多角度采集或上传本地照片，提取特征入库 |
| 管理模式 | 查看/编辑/删除用户，补充采集特征 |

## 活体检测原理

```
睁眼 → 瞳孔(暗) + 巩膜(亮) → 眼部区域像素方差高
闭眼 → 均匀肤色 → 方差骤降
眨眼 = 方差先降至峰值70%以下，再恢复
```

照片/视频攻击因缺乏自然眨眼波动模式而被判定为非活体。
