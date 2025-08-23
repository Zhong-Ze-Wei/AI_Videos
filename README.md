# Make_Videos_Mvp

## 项目概述
一个基于 FastAPI 和 Qwen API 的视频故事生成最小可行产品（MVP），包含前端和后端部分。

## 核心功能
1. **后端**:
   - 调用阿里云 Qwen API 生成视频故事 JSON 数据。
   - 提供 `/story` 接口，接收主题参数并返回故事数据。
   - 使用 FastAPI 框架构建，支持跨域请求。
2. **前端**:
   - 静态页面，动态加载后端生成的故事数据并渲染。

## 项目结构
```
Make_Videos_Mvp/
├── backend/                # 后端代码
│   ├── .env                # 环境变量配置（需配置 QWEN_API_KEY）
│   ├── main.py             # FastAPI 主程序入口
│   ├── models.py           # 数据模型定义（Story, Segment, Analytics）
│   ├── requirements.txt    # Python 依赖列表
│   ├── story_service.py    # Qwen API 调用逻辑
│   └── app/                # 子模块（可选）
└── frontend/               # 前端代码
    └── index.html          # 静态页面入口
```

## 技术实现
### 后端
- **FastAPI**: 提供 RESTful API 服务。
- **Qwen API**: 生成视频故事数据，包括标题、片段、风格分析等。
- **数据模型**: 使用 Pydantic 定义故事结构，确保数据格式一致。

### 前端
- **静态页面**: 使用原生 JavaScript 动态加载和渲染故事数据。

## 安装与运行

### 1. 后端
1. **安装依赖**:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. **配置环境变量**:
   - 在 `backend/.env` 中配置 `QWEN_API_KEY`（阿里云 Qwen API 密钥）。
3. **启动服务**:
   ```bash
   cd backend && python main.py
   ```
   - 服务默认运行在 `http://localhost:8000`。
4. 部署使用： 
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ngrok http 8000

   ```
### 2. 前端
1. **直接打开静态页面**:
   ```bash
   start frontend/index.html
   ```
   - 页面将在默认浏览器中打开，并自动加载默认主题的故事数据。

## 环境要求
- Python 3.7+
- FastAPI 和 Uvicorn（通过 `requirements.txt` 安装）

## 后续扩展
- 支持更多主题和自定义参数。
- 添加视频生成功能，结合故事数据生成实际视频。
- 优化前端交互和可视化效果。