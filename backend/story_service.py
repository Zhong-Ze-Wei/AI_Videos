import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError
from typing import List, Optional

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量获取 API Key
# 注意：请确保您的 .env 文件中是 DASHSCOPE_API_KEY="sk-xxxx"
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not QWEN_API_KEY:
    raise RuntimeError("❌ DASHSCOPE_API_KEY 没有配置，请检查 .env 文件或环境变量。")

# 初始化 OpenAI 客户端
# 这个客户端会复用，所以放在函数外面
try:
    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
except Exception as e:
    print(f"❌ OpenAI 客户端初始化失败: {e}")
    client = None


async def generate_story(theme: str):
    """
    使用 OpenAI 兼容模式调用 Qwen API，生成导演驾驶舱故事 JSON
    :param theme: 故事主题
    """
    if not client:
        print("❌ OpenAI 客户端未初始化，无法生成故事。")
        return None

    # 使用一组更通用的固定情感维度
    emotions = ["喜悦", "悲伤", "紧张", "轻松"]

    # 动态生成 emotion 字段的 JSON 示例字符串
    emotion_fields = ",\n".join([f'          "{emo}": 0' for emo in emotions])

    # 构造 Prompt，现在 emotion 部分是固定的
    prompt = f"""
你是一个导演驾驶舱数据生成器。
输入主题：{theme}

请严格按照以下格式输出一个 JSON，不要包含任何多余的说明性文本：
{{
  "title": "...",
  "segments": [
    {{
      "id": 1,
      "title": "...",
      "action": "...",
      "styleAnalysis": ["...", "..."],
      "firstFramePrompt": "...",
      "videoPrompt": "...",
      "imageUrl": "1.png",
      "analytics": {{
        "tension": 0,
        "complexity": 0,
        "pacing": 0,
        "emotion": {{
{emotion_fields}
        }}
      }}
    }}
  ]
}}
    """

    try:
        # 发起 API 调用
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )

        # 提取模型返回的文本内容
        text_content = completion.choices[0].message.content
        print(f"✅ AI 原始返回文本:\n{text_content}\n")

        # 清理并解析 JSON
        clean_text = re.sub(r"```(json)?", "", text_content).strip()
        story_json = json.loads(clean_text)

        return story_json

    except APIStatusError as e:
        # 处理 API 返回的错误，例如 401 (API Key错误), 429 (请求过载)
        print(f"❌ API 错误: {e.status_code}")
        print(f"错误信息: {e.response.text}")
        return None
    except APIConnectionError as e:
        # 处理网络连接错误
        print(f"❌ 网络连接错误: {e.__cause__}")
        return None
    except json.JSONDecodeError as e:
        # 处理 JSON 解析错误
        print(f"❌ JSON 解析错误: {e}")
        print(f"试图解析的文本：\n{clean_text}")
        return None
    except Exception as e:
        # 捕获其他未知错误
        print(f"❌ 发生未知错误: {e}")
        return None
