from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict
import os
import json
import asyncio
import re
import requests
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError
from http import HTTPStatus
from dashscope import ImageSynthesis

# --- 环境设置 ---
load_dotenv()
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not QWEN_API_KEY:
    raise RuntimeError("❌ DASHSCOPE_API_KEY 没有配置，请检查 .env 文件或环境变量。")

# --- AI 客户端初始化 ---
try:
    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
except Exception as e:
    print(f"❌ OpenAI 客户端初始化失败: {e}")
    client = None


# --- Pydantic 数据模型 ---
class BriefKeywords(BaseModel):
    style: str
    character: str
    world: str


class BriefDetails(BaseModel):
    style: str
    character: str
    world: str


class StoryRequest(BaseModel):
    brief: BriefDetails
    theme: str


# --- 修改点：为Analytics模型增加详细描述 ---
class Analytics(BaseModel):
    tension: float = Field(..., description="叙事张力指数 (0.0-10.0)")
    complexity: float = Field(..., description="视觉复杂度指数 (0.0-10.0)")
    pacing: float = Field(..., description="叙事节奏指数 (0.0-10.0)")
    emotion: Dict[str, float] = Field(..., description="多维度情感光谱指数 (0.0-10.0)")


class Segment(BaseModel):
    id: int
    title: str
    action: str
    analytics: Analytics
    firstFramePrompt: str
    videoPrompt: str


class Story(BaseModel):
    title: str
    segments: List[Segment]

class ImageGenerationRequest(BaseModel):
    brief: BriefDetails
    segments: List[Segment]

class ImageResponse(BaseModel):
    image_urls: List[str]


# --- FastAPI 应用实例 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI 服务 ---
async def generate_brief_from_ai(keywords: BriefKeywords) -> BriefDetails:
    """使用真实的AI模型根据关键词生成详细简报"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI 客户端未初始化")

    prompt = f"""
    你是一个创意总监，擅长将简洁的关键词扩展成富有想象力和细节的描述。
    根据用户提供的三个核心关键词，为一部短片生成详细的“创意简报”。
    请严格按照以下JSON格式返回，不要包含任何额外的说明性文本。

    用户输入：
    - 美术风格关键词: {keywords.style}  # 描述包括色调、光影效果、纹理和整体视觉风格
    - 角色描述关键词: {keywords.character}  # 角色的外貌、性格特征、动机、情感和背景故事
    - 世界观关键词: {keywords.world}  # 该世界的物理规则、社会结构、历史背景、技术与环境

    输出JSON格式：
    {{
      "style": "...",  # 描述风格，可能包括未来主义、复古、抽象、写实等风格的融合
      "character": "...",  # 描述角色的外貌、背景、内心动机和情感层次
      "world": "..."  # 解释该世界的文化、社会、科技以及自然环境等
    }}
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'user', 'content': prompt}],
            response_format={"type": "json_object"}  # 请求JSON输出
        )
        print(
            f"✅ 创意简报生成成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")

        response_text = completion.choices[0].message.content
        brief_data = json.loads(response_text)
        return BriefDetails(**brief_data)
    except (APIStatusError, APIConnectionError) as e:
        raise HTTPException(status_code=500, detail=f"AI 服务调用失败: {str(e)}")
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=500, detail=f"AI 返回格式错误: {str(e)}")


async def generate_story_from_ai(request: StoryRequest) -> Story:
    """使用真实的AI模型根据简报和主题生成完整故事，并带有优化的重试机制"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI 客户端未初始化")

    initial_prompt = f"""
    你是一位资深的电影编剧和导演，擅长将创意概念发展成结构完整、细节丰富的短片剧本。
    基于用户提供的“创意简报”和“故事主题”，请创作一个分为6个段落（幕）的完整故事剧本。
    每一幕都需要包含情节描述、数据分析以及用于AI绘画和视频生成的详细提示词。
    对于 "analytics" 中的 tension, complexity, pacing, 和 emotion 的所有值，请生成 0.0 到 10.0 之间的浮点数值。
    请严格按照以下JSON格式返回，不要包含任何额外的说明性文本。
    
    ---
    [创意简报]
    美术风格: {request.brief.style}
    角色描述: {request.brief.character}
    世界观: {request.brief.world}
    ---
    [故事主题]
    {request.theme}
    ---
    
    [输出JSON格式]
    {{
      "title": "一个关于'{request.theme}'的标题",
      "segments": [
        {{
          "id": 1,
          "title": "第一幕的标题",
          "action": "第一幕的详细情节描述，长度约50-80字。必须融合创意简报中的角色和世界观设定。",
          "analytics": {{
            "tension": 0.0, "complexity": 0.0, "pacing": 0.0,
            "emotion": {{ "喜悦": 0.0, "悲伤": 0.0, "紧张": 0.0, "轻松": 0.0 }}
          }},
          "firstFramePrompt": "一段详细的、用于生成静态图像的提示词。必须无缝融合[美术风格]、[角色描述]和本幕的[action]内容。风格要具体，画面描述要生动。",
          "videoPrompt": "一段详细的、用于生成8秒视频的提示词。在图像提示词的基础上，增加镜头运动（如：推、拉、摇、移）、角色动作、环境动态和整体节奏的描述。"
        }},
        {{ "id": 2, "title": "...", "action": "...", "analytics": {{...}}, "firstFramePrompt": "...", "videoPrompt": "..." }}
      ]
    }}
    """

    messages = [{'role': 'user', 'content': initial_prompt}]
    max_retries = 3

    for attempt in range(max_retries):
        try:
            print(f"正在进行第 {attempt + 1} 次AI调用...")
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                response_format={"type": "json_object"}
            )
            print(
                f"✅ 第 {attempt + 1} 次AI调用成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")

            response_text = completion.choices[0].message.content

            story_data = json.loads(response_text)
            story_object = Story(**story_data)

            print("AI返回格式正确，验证通过。")
            return story_object

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"第 {attempt + 1} 次尝试失败: AI返回格式错误。错误: {e}")
            if attempt < max_retries - 1:
                print("将在下一次尝试中要求AI修正格式。")
                messages.append({'role': 'assistant', 'content': response_text})
                messages.append({'role': 'user',
                                 'content': '你上次返回的文本无法被解析为JSON。请修正该文本，确保它是一个严格符合格式的JSON对象，不要添加任何额外的解释或注释。'})
            else:
                print("所有重试均失败，无法从AI获取有效数据。")
                raise HTTPException(status_code=500,
                                    detail=f"AI 返回格式或内容不符合要求，经过多次尝试后仍然失败: {str(e)}")
        except (APIStatusError, APIConnectionError) as e:
            raise HTTPException(status_code=500, detail=f"AI 服务调用失败: {str(e)}")

    raise HTTPException(status_code=500, detail="AI 服务未知错误，重试循环结束但未返回结果。")


# --- 优化后的图像生成函数 ---
def call_wanx_sync_batch(prompt: str, api_key: str, num_images: int, segment_title: str) -> List[str]:
    """同步调用通义万相，生成图片，保存到本地，并返回URL列表。"""
    print(f"  - 正在为Prompt生成 {num_images} 张图片...")
    rsp = ImageSynthesis.call(
        api_key=api_key,
        model='wanx2.1-t2i-plus',
        prompt=prompt,
        n=num_images,
        size='1280*720',
        parameters={'prompt_extend': 'false'}
    )
    if rsp.status_code == HTTPStatus.OK:
        urls = [result.url for result in rsp.output.results]
        print(f"  - 成功生成 {len(urls)} 张图片。")

        # --- 新增：保存图片到本地 ---
        save_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(save_dir, exist_ok=True)

        # 清理文件名中的非法字符
        safe_title = re.sub(r'[\\/*?:"<>|]', "", segment_title)

        for i, url in enumerate(urls):
            try:
                response = requests.get(url)
                response.raise_for_status()  # 确保请求成功
                # 构建文件名，例如 "第一幕_故事的转折_1.png"
                file_name = f"{safe_title}_{i + 1}.png"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"  - 图片已保存到: {file_path}")
            except requests.RequestException as e:
                print(f"  - 下载图片失败: {url}, 错误: {e}")

        return urls
    else:
        print(f"  - 通义万相调用失败, status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}")
        return [f"https://placehold.co/400x225/FF0000/FFFFFF?text=Error:{rsp.code}" for _ in range(num_images)]


async def generate_images_from_ai(request: ImageGenerationRequest) -> ImageResponse:
    """使用真实的通义万相模型并行生成图片，并使用Semaphore控制并发。"""
    semaphore = asyncio.Semaphore(2)
    tasks = []

    async def controlled_generation(segment):
        """一个包装函数，用于在Semaphore的控制下执行生成任务"""
        async with semaphore:
            print(f"  -> 开始为第 {segment.id} 幕 '{segment.title}' 生成图片...")
            combined_prompt = f"{request.brief.style}, {request.brief.character}, {request.brief.world}, {segment.firstFramePrompt}"

            # --- 修改点 1：暂时只生成一张图片 ---
            # 未来可以改回 2
            num_to_generate = 1

            urls = await asyncio.to_thread(call_wanx_sync_batch, combined_prompt, QWEN_API_KEY, num_to_generate,
                                           segment.title)

            # --- 修改点 2：增加休眠时间 ---
            print(f"  -> 第 {segment.id} 幕生成完毕，休眠1秒...")
            await asyncio.sleep(1)
            return urls

    for segment in request.segments:
        tasks.append(controlled_generation(segment))

    print(f"正在为 {len(tasks)} 个段落生成图片 (并发限制: 2)...")
    try:
        results_list_of_lists = await asyncio.gather(*tasks)
        all_urls = [url for sublist in results_list_of_lists for url in sublist]
        print(f"✅ 所有图片生成完成，共 {len(all_urls)} 张。")
        return ImageResponse(image_urls=all_urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像生成过程中发生错误: {str(e)}")


# --- API 接口 ---
@app.post("/generate_brief", response_model=BriefDetails)
async def generate_brief(keywords: BriefKeywords):
    return await generate_brief_from_ai(keywords)


@app.post("/generate_story_details", response_model=Story)
async def generate_story_details(request: StoryRequest):
    return await generate_story_from_ai(request)


@app.post("/generate_images", response_model=ImageResponse)
async def generate_images(request: ImageGenerationRequest):
    return await generate_images_from_ai(request)


# 用于直接访问前端的根路径
from fastapi.responses import FileResponse


@app.get("/", response_class=FileResponse)
async def read_index():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if not os.path.exists(frontend_path):
        return {"error": "index.html not found"}
    return FileResponse(frontend_path)
