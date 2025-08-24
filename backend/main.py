from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, AsyncGenerator
import os
import json
import asyncio
import re
from fastapi.staticfiles import StaticFiles
import glob # 用于文件查找
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
    preview_image_url: str | None = None # 新增此行

# --- FastAPI 应用实例 ---
app = FastAPI()
# 挂载静态文件目录，允许通过 /static/data/... 的 URL 访问 ../data/ 文件夹下的内容
app.mount("/static/data", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "data")), name="data")

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
    使用“两步走”策略生成故事，并为每一幕动态计算分析指数。
    第一步：生成包含标题和六幕剧情梗概的整体故事大纲。
    第二步：对每一幕，先进行创意扩写，然后基于扩写内容进行独立的数据分析。
    """
    if not client:
        raise StopAsyncIteration

    # --- 第一步：生成宏观的故事大纲和标题 ---
    try:
        print("--- 正在生成整体故事大纲 ---")
        overall_plot_prompt = f"""
        你是一位顶级的电影故事构思师。请基于用户提供的“创意简报”和“故事主题”，创作一个完整、连贯、包含六个幕的短片故事大纲。

        [创意简报]
        - 美术风格: {request.brief.style}
        - 角色描述: {request.brief.character}
        - 核心主题: {request.brief.world}

        [故事主题]
        {request.theme}

        [你的任务]
        1. 为这个故事起一个富有吸引力的标题。
        2. 将故事分为清晰的六幕（Act 1 到 Act 6），每一幕都用80-120字详细描述其核心剧情，确保六幕剧情能够连贯地组成一个有起因、发展、高潮和结局的完整故事。

        [输出格式]
        请严格按照以下JSON格式返回，不要包含任何额外的说明性文本：
        {{
          "title": "你的故事标题",
          "plot_outline": [
            "第一幕的详细剧情梗概...",
            "第二幕的详细剧情梗概...",
            "第三幕的详细剧情梗概...",
            "第四幕的详细剧情梗概...",
            "第五幕的详细剧情梗概...",
            "第六幕的详细剧情梗概..."
          ]
        }}
        """
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'user', 'content': overall_plot_prompt}],
            response_format={"type": "json_object"}
        )
        print(f"✅ 整体故事大纲生成成功 - Token 消耗: {completion.usage.total_tokens}")

        story_structure = json.loads(completion.choices[0].message.content)
        title = story_structure.get("title", f"关于“{request.theme}”的故事")
        plot_outline = story_structure.get("plot_outline", [])

        if len(plot_outline) != 6:
            raise ValueError("AI未能生成包含六幕的剧情大纲。")

        yield json.dumps({"title": title}) + "\n"

    except Exception as e:
        print(f"❌ 关键步骤失败：整体故事大纲生成失败: {e}")
        yield json.dumps({"error": f"故事大纲生成失败: {e}"}) + "\n"
        raise StopAsyncIteration

    # --- 第二步：基于大纲，循环生成每一幕的详细信息 ---
    previous_actions = []
    for i in range(1, 7):
        context = "\n".join(f"- 第{idx + 1}幕: {act}" for idx, act in enumerate(previous_actions))
        current_plot_point = plot_outline[i - 1]

        segment_data = {}
        try:
            # --- 步骤 2a: 创意内容生成 ---
            print(f"--- 正在为第 {i} 幕生成创意内容 ---")
            creative_prompt = f"""
            你是一位资深的电影编剧和分镜师。请专注于创作故事的【第 {i} 幕】。

            [本幕核心剧情 - 必须严格遵循]
            {current_plot_point}

            [你的任务]
            请严格围绕上方指定的“本幕核心剧情”，将其扩写成一个包含创意内容的JSON对象。
            `action`字段必须是对“本幕核心剧情”的生动、详细的再创作。

            [输出JSON格式]
            请严格按照以下JSON格式返回，不要包含任何与分析相关的字段：
            {{
              "id": {i},
              "title": "第{i}幕的简短标题",
              "action": "对‘本幕核心剧情’进行扩写后的详细情节描述，长度约80-120字...",
              "firstFramePrompt": "一段详细的、用于生成静态图像的提示词，必须生动地展现本幕的action内容和核心视觉...",
              "videoPrompt": "一段详细的、用于生成8秒视频的提示词，在图像提示词基础上增加镜头运动、角色动作和环境动态描述..."
            }}
            """
            creative_completion = client.chat.completions.create(
                model="qwen-plus", messages=[{'role': 'user', 'content': creative_prompt}],
                response_format={"type": "json_object"}
            )
            creative_data = json.loads(creative_completion.choices[0].message.content)
            segment_data.update(creative_data)
            print(f"✅ 第 {i} 幕创意内容生成成功。")

            # --- 步骤 2b: 基于 action 内容进行数据分析 ---
            print(f"--- 正在为第 {i} 幕计算分析指数 ---")
            action_text_to_analyze = creative_data.get('action', '')
            analytics_prompt = f"""
            你是一位专业的叙事数据分析师。请分析以下这段剧情描述，并为其量化各项指数。

            [剧情描述]
            {action_text_to_analyze}

            [分析维度定义]
            - tension (叙事张力): 剧情的悬念、冲突和紧张程度。0为平淡，10为极度紧张。
            - complexity (视觉复杂度): 画面中可能出现的元素、细节和构图的复杂性。0为极简，10为极其复杂。
            - pacing (叙事节奏): 剧情推进的速度感。0为缓慢/静态，10为极快/蒙太奇。
            - emotion (情感光谱): 剧情所蕴含的核心情感强度，各项之和不必为10。

            [输出任务]
            请严格按照以下JSON格式返回分析结果，数值必须是浮点数：
            {{
                "tension": 0.0,
                "complexity": 0.0,
                "pacing": 0.0,
                "emotion": {{
                    "喜悦": 0.0,
                    "悲伤": 0.0,
                    "愤怒": 0.0,
                    "恐惧": 0.0,
                    "惊讶": 0.0,
                    "紧张": 0.0
                }}
            }}
            """
            analytics_completion = client.chat.completions.create(
                model="qwen-plus", messages=[{'role': 'user', 'content': analytics_prompt}],
                response_format={"type": "json_object"}
            )
            analytics_data = json.loads(analytics_completion.choices[0].message.content)
            segment_data['analytics'] = analytics_data
            print(f"✅ 第 {i} 幕分析指数计算成功。")

            # --- 合并、验证并输出 ---
            Segment(**segment_data)  # 使用Pydantic模型验证最终合并的数据结构
            previous_actions.append(segment_data['action'])
            yield json.dumps(segment_data) + "\n"

        except (ValidationError, json.JSONDecodeError, KeyError) as e:
            print(f"❌ 第 {i} 幕生成过程中失败: {e}")
            yield json.dumps({"error": f"第 {i} 幕生成失败: {e}"}) + "\n"
            # 即使失败，也跳到下一幕，避免整个流程中断
            continue
        except Exception as e:
            yield json.dumps({"error": f"AI服务调用失败: {e}"}) + "\n"
            raise StopAsyncIteration

# 基于前端传回的背景信息和提示词生成AI绘画的提示词
async def refine_prompt_with_ai(brief: BriefDetails, segment_prompt: str) -> str:
    """
    使用“一致性锚点注入法”的全新策略，将全局设定和单幕Prompt融合成一个高度优化的最终Prompt。
    这个新方法旨在最大化跨图像的角色和风格一致性。
    """
    if not client:
        # 如果AI客户端不可用，则退回基础拼接模式
        return f"{segment_prompt}, {brief.style}, {brief.character}"

    # --- 全新的、更强大的提示词工程策略 ---
    refiner_prompt = f"""
        你是一位世界顶级的AI绘画提示词（Prompt）架构师，尤其擅长为系列图像创建能保持高度视觉一致性的“母版提示词”。
        你的任务是基于一个“全局一致性简报”和一个“特定场景描述”，构建一个单一、完整、且高度优化的最终提示词字符串。
        ---
        [全局一致性简报 (GLOBAL CONSISTENCY BRIEF)]
        # 美术风格: {brief.style}
        # 角色描述: {brief.character}
        # 核心主题: {brief.world}
        ---
        [特定场景描述 (SPECIFIC SCENE PROMPT)]
        {segment_prompt}
        ---
        [你的执行步骤 (MANDATORY EXECUTION STEPS)]
        1. **识别并提取“一致性锚点”**:
            - **角色锚点 (Character Anchor)**: 从“角色描述”的“关键词:”部分，提取出所有描述核心角色外貌（例如：“旺泽”）和固定装备的关键词。确保这些描述细致、具体，并在每一幕的提示词中重复使用。每次生成时，角色的外观、服装、发型、表情、动作等都必须与上一幕保持一致。
            - **风格锚点 (Style Anchor)**: 从“美术风格”和“核心主题”的“关键词:”部分，提取出定义世界观和视觉风格的关键词。强调每个场景的光影效果、颜色调性、细节层次等视觉元素，以确保所有场景在风格上的一致性。
        2. **构建最终提示词结构**:
            - **主体场景**: 将“特定场景描述”作为核心，生动地描述这一幕中发生的具体事件、环境和人物的动作，同时确保这些描述与全局一致性简报中的角色和风格信息一致。
            - **角色锚点注入**: 在主体场景描述之后，**必须原封不动地、完整地**插入你提取的“角色锚点”短语。例如：“一只名叫旺泽的短毛柯基犬，头戴镶有银色符文的黑色侠士斗笠，戴着黑框研究生眼镜，颈部佩戴高科技金属项圈，金棕色毛发，情绪激动时尾巴发出橙色光芒”。
            - **风格与世界观锚点**: 附加上你提取的“风格锚点”，例如：“东方水墨画风，玄幻未来主义，光影对比强烈，颠倒社会，犬类文明”。
        3. **生成提示词时的关键注意点**:
            - **风格一致性**：每个场景的视觉元素（如光影、色调、背景元素等）应保持一致，避免大幅变化，特别是背景中的光照、色彩以及使用的纹理。
            - **角色一致性**：角色的外观、装备、动作、情绪等必须在各个场景中保持一致。例如，如果角色戴着特定的帽子或衣物，必须在每个场景的描述中反复提到这些细节，避免遗漏。
        [输出要求]
        - **绝对禁止**任何形式的解释、标题或前言。
        - **必须**使用中文进行输出。
        - 最终结果必须是一个**单一的、由逗号连接的字符串**。
        [示例]
        假设“特定场景描述”是“旺泽在一个充满未来感的弄堂里，警惕地发现了一张发光的卷轴”。
        一个优秀的输出应该是：
        "在一个充满未来科技感的潮湿弄堂里，一只柯基犬警惕地凝视着地上发光的卷轴, **一只名叫旺泽的短毛柯基犬，头戴镶有银色符文的黑色侠士斗笠，戴着黑框研究生眼镜，颈部佩戴高科技金属项圈，金棕色毛发，情绪激动时尾巴发出橙色光芒**, 东方水墨画风, 玄幻未来主义, 光影对比强烈, 颠倒社会, 犬类文明"
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'user', 'content': refiner_prompt}]
        )
        print(f"    - 提示词精炼成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")
        refined_prompt = completion.choices[0].message.content.strip()
        # 确保返回的是单行文本
        return refined_prompt.replace('\n', ', ').replace('\r', '')
    except Exception as e:
        print(f"    - 提示词精炼失败，将使用原始提示词。错误: {e}")
        # 在失败时，也尝试使用关键词进行拼接
        style_keywords = brief.style.split('关键词:')[1].split('\n')[0].strip() if '关键词:' in brief.style else ''
        char_keywords = brief.character.split('关键词:')[1].split('\n')[0].strip() if '关键词:' in brief.character else ''
        return f"{segment_prompt}, {char_keywords}, {style_keywords}"

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
            try:
                final_prompt = await refine_prompt_with_ai(request.brief, segment.firstFramePrompt)
                print(f"    - 最终生成提示词 (ID: {segment.id}): {final_prompt}")

                safe_title = re.sub(r'[\\/*?:"<>|]', "", segment.title)
                file_name = f"{segment.id:02d}_{safe_title}.png"
                file_path = os.path.join(save_directory, file_name)

                url = await asyncio.to_thread(call_wanx_sync_and_save, final_prompt, QWEN_API_KEY, file_path)

                # 检查返回的 URL 是否是错误占位符
                if "placehold.co" in url or "Error" in url:
                    # 如果是错误，我们主动抛出一个异常，以便在 gather 中被捕获
                    raise Exception(f"图片生成或下载失败 (ID: {segment.id})")

                await asyncio.sleep(1)
                return {"final_prompt": final_prompt, "url": url}
            except Exception as e:
                # 如果单个任务内部发生任何错误，捕获它并返回一个错误对象
                # 这可以防止 gather 中断
                print(f"❌ 图片生成任务失败 (ID: {segment.id}): {e}")
                # 返回一个可识别的错误结果，而不是让异常传播出去
                return e

    for segment in request.segments:
        tasks.append(controlled_generation(segment))

    print(f"正在为 {len(tasks)} 个段落生成图片 (并发限制: 2)...")

    # --- 关键修改：添加 return_exceptions=True ---
    # 这将使 gather 等待所有任务完成，并将异常作为结果返回，而不是直接抛出
    all_results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"✅ 所有图片生成任务已执行完毕。")

    # --- 后续处理，分离成功和失败的结果 ---
    successful_results = []
    failed_tasks_count = 0
    for result in all_results_or_errors:
        if isinstance(result, Exception):
            # 如果结果是一个异常对象，说明这个任务失败了
            failed_tasks_count += 1
            # 我们可以选择在这里记录更详细的日志
        else:
            # 否则，这是一个成功的结果
            successful_results.append(result)

    if failed_tasks_count > 0:
        print(f"⚠️  {failed_tasks_count} 个图片生成任务失败。")

    if not successful_results:
        # 如果没有任何图片成功，则直接抛出异常
        raise HTTPException(status_code=500, detail="所有图片生成任务均失败，请检查后端日志。")

    # --- 即使部分失败，也继续保存项目 ---
    try:
        project_data = {
            "brief": request.brief.dict(),
            "story": {"title": request.title, "segments": [s.dict() for s in request.segments]},
            # 注意：这里我们只保存成功的结果
            "images": {"image_results": successful_results}
        }
        project_file_path = os.path.join(save_directory, "project_data.json")
        with open(project_file_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 项目数据已保存到: {project_file_path}")

        # 返回给前端的也只有成功的图片URL
        image_urls = [result['url'] for result in successful_results]
        return ImageResponse(image_urls=image_urls)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像生成后，项目保存过程中发生错误: {str(e)}")

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
    """列出所有已保存的项目，并为每个项目附加一张预览图URL"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.exists(data_dir):
        return []

    tasks = []
    for folder_name in sorted(os.listdir(data_dir), reverse=True):  # 按名称排序，新的在前
        task_folder_path = os.path.join(data_dir, folder_name)
        project_file = os.path.join(task_folder_path, "project_data.json")

        if os.path.isdir(task_folder_path) and os.path.exists(project_file):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    title = data.get("story", {}).get("title", folder_name)

                    # --- 新增：查找预览图 ---
                    preview_image_url = None
                    # 查找文件夹下第一个 png 或 jpg 文件
                    image_files = glob.glob(os.path.join(task_folder_path, "*.png")) + glob.glob(
                        os.path.join(task_folder_path, "*.jpg"))
                    if image_files:
                        # 获取文件名并构建URL
                        image_name = os.path.basename(image_files[0])
                        preview_image_url = f"/static/data/{folder_name}/{image_name}"
                    # -----------------------

                    tasks.append(
                        TaskInfo(
                            folder_name=folder_name,
                            title=title,
                            preview_image_url=preview_image_url  # 附加URL
                        )
                    )
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