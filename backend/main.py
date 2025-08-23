from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, AsyncGenerator
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
    title: str  # 增加 title 字段以匹配前端请求
    brief: BriefDetails
    segments: List[Segment]

class ImageResponse(BaseModel):
    image_urls: List[str]

class ProjectData(BaseModel):
    brief: BriefDetails
    story: Story
    images: ImageResponse

class TaskInfo(BaseModel):
    folder_name: str
    title: str

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
# 创建背景
async def generate_brief_from_ai(keywords: BriefKeywords) -> BriefDetails:
    """使用真实的AI模型根据关键词生成详细简报"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI 客户端未初始化")

    prompt = f"""
        你是一个创意总监，擅长将简洁的关键词扩展成富有想象力和细节的描述。
        根据用户提供的三个核心关键词，为一部短片生成详细的“创意简报”。
        在“style”描述的最前，必须明确提炼出3-5个用于在AI绘画中保持风格高度一致的“核心一致性关键词”，并用 "关键词:" 开头。
        在“character”描述的最前，必须明确提炼出5-10个用于在AI绘画中保持人物形象高度一致的“核心关键词”，并用 "关键词:" 开头，必须保证有异常详细，具备特征的人物细节描写，禁止用模糊性形容词
        在“world”描述的最前，必须明确提炼出3-5个用于在AI绘画中保持故事核心主题的“核心一致性关键词”，并用 "关键词:" 开头。
        请严格按照以下JSON格式返回，不要包含任何额外的说明性文本。
    
        用户输入：
        - 美术风格关键词: {keywords.style} 
        - 角色描述关键词: {keywords.character}  
        - 核心主题关键词: {keywords.world} 
    
        输出JSON格式：
        {{
          "style": "...",  # 关键词，换行后输出细节，描述包括色调、光影效果、纹理和整体视觉风格，可能包括未来主义、复古、抽象、写实等风格的融合
          "character": "...",  # 关键词，换行后输出细节，角色的超级细节外貌、性格特征、动机、情感和背景故事
          "world": "..."  # 关键词，换行后输出细节，该故事的核心情感，节奏张力，该世界的视觉氛围、情感波动、节奏感、画面冲击力，适应短视频流行的结构和易消费风格
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


# 故事生成
async def generate_story_from_ai(request: StoryRequest) -> AsyncGenerator[str, None]:
    """
    使用真实的AI模型，通过逐幕生成的方式构建完整故事，以提高稳定性和连贯性。
    这是一个异步生成器，会逐个yield出JSON字符串。
    """
    if not client:
        raise StopAsyncIteration

    # 第一步：先独立生成一个总标题
    try:
        title_prompt = f"为故事主题 '{request.theme}' 起一个富有吸引力的标题，并以JSON格式返回：{{\"title\": \"...\"}}"
        completion = client.chat.completions.create(
            model="qwen-plus", messages=[{'role': 'user', 'content': title_prompt}], response_format={"type": "json_object"}
        )
        title_data = json.loads(completion.choices[0].message.content)
        # 流式输出第一条数据：标题
        yield json.dumps(title_data) + "\n"
    except Exception as e:
        print(f"⚠️ 故事总标题生成失败: {e}")
        yield json.dumps({"title": f"关于“{request.theme}”的故事"}) + "\n"

    # 第二步：循环生成6个独立的幕
    previous_actions = []
    for i in range(1, 7):
        context = "\n".join(f"- 第{idx+1}幕: {act}" for idx, act in enumerate(previous_actions))

        segment_prompt = f"""
                你是一位资深的电影编剧。基于“创意简报”、“故事主题”和“前情提要”，请仅创作【第 {i} 幕】的内容。
                需要包含情节描述、数据分析以及用于AI绘画和视频生成的超级详细提示词，其中...
                [你的任务]
                请严格按照以下JSON格式返回【第 {i} 幕】的内容，不要包含任何额外文本：
                {{
                  "id": {i},
                  "title": "第{i}幕的标题", 
                  "action": "第{i}幕的详细情节描述，长度约80-120字，必须承接前情提要。例如：主角在一个阴暗的巷子里发现了线索，随后追踪到一个神秘的组织，然而突然间遭遇了敌人的伏击……", 
                  "analytics": {{
                    "tension": 7.5,  # 0到10之间，表示该幕的情节张力
                    "complexity": 6.2,  # 0到10之间，表示该幕的情节复杂度
                    "pacing": 5.0,  # 0到10之间，表示该幕的叙事节奏
                    "emotion": {{
                      "喜悦": 1.0,  # 0到10之间，表示情感的强度
                      "悲伤": 3.5,  # 0到10之间，表示情感的强度
                      "紧张": 7.0,  # 0到10之间，表示情感的强度
                      "轻松": 0.5  # 0到10之间，表示情感的强度
                    }}
                  }},
                  "firstFramePrompt": "一段详细的、用于生成静态图像的提示词。基于本幕的[action]内容，突出本幕全部内容的特色。例如：画面中可以看到一条阴暗的巷子，湿漉漉的地面反射着昏黄的路灯光，主角正在走过，眼神充满警觉。巷子的两侧是破旧的建筑，远处传来隐约的脚步声……", 
                  "videoPrompt": "一段详细的、用于生成8秒视频的提示词。基于本幕的[action]内容，在图像提示词的基础上，增加镜头运动（如：推、拉、摇、移）、角色动作、环境动态和整体节奏的描述。例如：镜头从主角的背后拉近，逐渐转向侧面，捕捉到他紧张的表情。地面上的水滴开始闪烁反射，远处突然有一辆黑色车子驶入画面，节奏越来越紧张，音乐逐渐加快，镜头快速跟随主角奔跑……"
                }}
                """
        messages = [{'role': 'user', 'content': segment_prompt}]
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                print(f"--- 正在生成第 {i} 幕 (第 {attempt + 1} 次尝试) ---")
                completion = client.chat.completions.create(
                    model="qwen-plus", messages=messages, response_format={"type": "json_object"}
                )
                print(f"✅ 第 {i} 幕生成成功 - Token 消耗: {completion.usage.total_tokens}")
                segment_data = json.loads(completion.choices[0].message.content)
                Segment(**segment_data) # 验证格式
                previous_actions.append(segment_data['action'])
                # 流式输出这个片段
                yield json.dumps(segment_data) + "\n"
                success = True
                break
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"第 {i} 幕第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
                    messages.append({'role': 'user', 'content': '你上次返回的文本格式错误，请严格修正并重新生成。'})
            except Exception as e:
                yield json.dumps({"error": f"AI服务调用失败: {e}"}) + "\n"
                raise StopAsyncIteration
        if not success:
            yield json.dumps({"error": f"第 {i} 幕生成失败，已达最大重试次数。"}) + "\n"
            raise StopAsyncIteration


# 基于前端传回的背景信息和提示词生成AI绘画的提示词
async def refine_prompt_with_ai(brief: BriefDetails, segment_prompt: str) -> str:
    """使用AI将全局设定和单幕Prompt融合成一个优化的最终Prompt。"""
    if not client:
        return f"{brief.style}, {brief.character}, {brief.world}, {segment_prompt}"

    refiner_prompt = f"""
        你是一位顶级的AI绘画提示词工程师。你的任务是将用户提供的“核心主体”和“参考信息”融合成一个单一、强大、高度优化的提示词字符串，供文生图模型（如Midjourney, Stable Diffusion）使用。
        
        [参考信息]
        - 美术风格: {brief.style}
        - 角色描述: {brief.character}
        - 核心主题: {brief.world}
        
        [核心主体]
        {segment_prompt}
        
        请遵循以下规则进行融合与优化：
        1.  **语言一致性是最高优先级**: 必须严格使用与[核心主体]相同的语言（中文）来生成最终的提示词。禁止将内容翻译成英文或任何其他语言。
        2.  **人物一致性**: 必须严格、完整地使用[角色描述]中的“关键词”来描述角色，这是确保人物在所有画面中保持绝对一致的关键，不可省略或修改。
        3.  **主体优先**: 最终提示词必须以“核心主体”的内容为绝对核心，清晰地描述画面中的主要人物、动作和场景，适配参考信息中的风格。
        4.  **注入风格**: 将“参考信息”中的风格、角色和主题的关键描述词（特别是“关键词:”后面的部分）智能地、无缝地融入到核心主体的描述中，以确保视觉一致性。
        5.  **精炼简洁**: 最终输出的必须是一个单一的字符串，不要包含任何解释、标题或JSON格式。语言要精炼，删除冗余信息。
        6.  **结构优化**: 按照 "主体描述, 细节补充, 风格和技术性词语" 的结构组织最终的提示词。
        
        请直接输出优化后的最终提示词字符串。
        """
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'user', 'content': refiner_prompt}]
        )
        # --- 关键修改：增加Token消耗日志 ---
        print(f"    - 提示词精炼成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")
        refined_prompt = completion.choices[0].message.content.strip()
        return refined_prompt
    except Exception as e:
        print(f"    - 提示词精炼失败，将使用原始提示词。错误: {e}")
        return f"*核心*：{segment_prompt}, *背景参考*：{brief.style}, {brief.character}, {brief.world} "


# 保存生成图像到本地
def get_unique_save_directory(base_name: str) -> str:
    """根据基础名称创建唯一的任务文件夹路径。"""
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, "..", "data")

    safe_base_name = re.sub(r'[\\/*?:"<>|]', "", base_name).strip()
    if not safe_base_name:
        safe_base_name = "Untitled_Task"

    task_dir = os.path.join(data_dir, safe_base_name)

    counter = 1
    while os.path.exists(task_dir):
        task_dir = os.path.join(data_dir, f"{safe_base_name}-{counter}")
        counter += 1

    os.makedirs(task_dir, exist_ok=True)
    print(f"创建任务文件夹: {task_dir}")
    return task_dir


# 使用qwenapi生成图像
def call_wanx_sync_and_save(prompt: str, api_key: str, save_path: str) -> str:
    """调用通义万相，生成单张图片，保存到本地，并返回临时URL。"""
    print(f"  - 正在为 '{os.path.basename(save_path)}' 生成图片...")
    rsp = ImageSynthesis.call(
        api_key=api_key, model='wanx2.1-t2i-plus', prompt=prompt, n=1,
        size='1280*720', parameters={'prompt_extend': 'false','seed': 990527}
    )
    if rsp.status_code == HTTPStatus.OK:
        url = rsp.output.results[0].url
        print(f"  - 成功生成图片。")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  - 图片已保存到: {save_path}")
            return url
        except requests.RequestException as e:
            print(f"  - 下载图片失败: {url}, 错误: {e}")
            return f"https://placehold.co/400x225/FF0000/FFFFFF?text=DownloadFailed"
    else:
        print(f"  - 通义万相调用失败, code: {rsp.code}, message: {rsp.message}")
        return f"https://placehold.co/400x225/FF0000/FFFFFF?text=Error:{rsp.code}"


# 基于合成的提示词生成图像
async def generate_images_from_ai(request: ImageGenerationRequest) -> ImageResponse:
    semaphore = asyncio.Semaphore(2)
    tasks = []
    save_directory = get_unique_save_directory(request.title)

    async def controlled_generation(segment):
        async with semaphore:
            final_prompt = await refine_prompt_with_ai(request.brief, segment.firstFramePrompt)
            print(f"    - 最终生成提示词: {final_prompt}")
            safe_title = re.sub(r'[\\/*?:"<>|]', "", segment.title)
            file_name = f"{segment.id:02d}_{safe_title}.png"
            file_path = os.path.join(save_directory, file_name)
            url = await asyncio.to_thread(call_wanx_sync_and_save, final_prompt, QWEN_API_KEY, file_path)
            await asyncio.sleep(1)
            return {"final_prompt": final_prompt, "url": url}

    for segment in request.segments:
        tasks.append(controlled_generation(segment))

    print(f"正在为 {len(tasks)} 个段落生成图片 (并发限制: 2)...")
    try:
        all_results = await asyncio.gather(*tasks) # 这里只调用一次

        print(f"✅ 所有图片生成完成，共 {len(all_results)} 张。")

        project_data = {
            "brief": request.brief.dict(),
            "story": {"title": request.title, "segments": [s.dict() for s in request.segments]},
            "images": {"image_results": all_results}
        }
        project_file_path = os.path.join(save_directory, "project_data.json")
        with open(project_file_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 项目数据已保存到: {project_file_path}")

        image_urls = [result['url'] for result in all_results]
        return ImageResponse(image_urls=image_urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像生成或项目保存过程中发生错误: {str(e)}")


# --- API 接口 ---
@app.post("/generate_brief", response_model=BriefDetails)
async def generate_brief(keywords: BriefKeywords):
    return await generate_brief_from_ai(keywords)

@app.post("/generate_story_details")
async def generate_story_details(request: StoryRequest):
    return StreamingResponse(generate_story_from_ai(request), media_type="application/x-ndjson")

@app.post("/generate_images", response_model=ImageResponse)
async def generate_images(request: ImageGenerationRequest):
    return await generate_images_from_ai(request)

# --- 新增：项目管理接口 ---
@app.get("/tasks", response_model=List[TaskInfo])
async def list_tasks():
    """列出所有已保存的项目"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.exists(data_dir):
        return []

    tasks = []
    for folder_name in os.listdir(data_dir):
        project_file = os.path.join(data_dir, folder_name, "project_data.json")
        if os.path.isdir(os.path.join(data_dir, folder_name)) and os.path.exists(project_file):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    tasks.append(
                        TaskInfo(folder_name=folder_name, title=data.get("story", {}).get("title", folder_name)))
            except Exception as e:
                print(f"读取项目失败 {folder_name}: {e}")
    return tasks


@app.get("/task/{task_name}", response_model=ProjectData)
async def load_task(task_name: str):
    """加载指定的项目数据，同时处理旧数据格式"""
    project_file = os.path.join(os.path.dirname(__file__), "..", "data", task_name, "project_data.json")
    if not os.path.exists(project_file):
        raise HTTPException(status_code=404, detail="项目文件未找到")
    try:
        with open(project_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # ====== 修正旧数据格式：同时处理键名和数据类型 ======
            # 1. 检查是否存在 'images' 字段
            if 'images' in data:
                images_data = data['images']
                # 2. 检查旧的'image_results'字段，如果存在则进行转换
                if 'image_results' in images_data:
                    old_results = images_data.pop('image_results')
                    # 3. 将字典列表转换为字符串列表
                    new_urls = []
                    for item in old_results:
                        if isinstance(item, dict) and 'url' in item:
                            new_urls.append(item['url'])
                    images_data['image_urls'] = new_urls
            # ==================================================

            return ProjectData(**data)
    except Exception as e:
        # 这会捕获所有可能的文件I/O或JSON解析错误
        raise HTTPException(status_code=500, detail=f"加载项目数据失败: {e}")

# 用于直接访问前端的根路径
@app.get("/", response_class=FileResponse)
async def read_index():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if not os.path.exists(frontend_path):
        return {"error": "index.html not found"}
    return FileResponse(frontend_path)