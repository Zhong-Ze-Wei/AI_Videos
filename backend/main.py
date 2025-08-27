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
import glob  # 用于文件查找
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

# ====================================================================
# 模型选择代码汇总
#
# 此部分列出了脚本中所有API调用的模型选择，便于统一管理和调整。
# ====================================================================

# 高级模型
# 1. 创意简报生成 (step1)作用: 创意总监角色，将关键词扩展为详细的简报。对应代码位置: generate_brief_from_ai()
# 2. 故事大纲生成 (step2 - 第一步)作用: 电影故事构思师，生成六幕故事大纲。对应代码位置: generate_story_from_ai() -> overall_plot_prompt
# 4. 角色图谱生成 (step2 - 第二步)作用: 角色设定师，将人物描述为可用于AI绘画的详细模板。对应代码位置: generate_story_from_ai() -> character_prompt
# 5. 单幕创意内容生成 (step2 - 第三步)作用: 资深编剧和分镜师，将大纲扩写为详细情节。对应代码位置: generate_story_from_ai() -> creative_prompt
# 7. AI绘画提示词精炼 (step3.1)作用: 提示词架构师，融合风格、场景和角色生成最终提示词。对应代码位置: refine_prompt_with_ai()
model_for_hard = "qwen-plus-2025-07-28"     #qwen-plus-2025-07-14   qwen-plus-2025-04-28

# 简单模型
# 3. 核心人物筛查 (step2 - 第二步)作用: 快速检索小助手，判断并提取核心人物。对应代码位置: generate_story_from_ai() -> core_detector_prompt
# 6. 数据分析与量化 (step2 - 第三步)作用: 专业数据分析师，为剧情量化各项指数。对应代码位置: generate_story_from_ai() -> analytics_prompt
model_for_easy = "qwen-turbo"     #"qwen-turbo"


# 8. AI图像生成 (step3.2)作用: 图像生成模型，根据提示词生成图片。对应代码位置: call_wanx_sync_and_save()
model_for_image = "wan2.2-t2i-plus"       # "wanx2.1-t2i-plus" "wan2.2-t2i-plus" "qwen-image"

# ====================================================================

# 在您的代码中，您可以将上述模型变量赋值到相应的位置，例如：
# completion = client.chat.completions.create(
#     model=model_for_brief_generation,
#     messages=[{'role': 'user', 'content': prompt}],
#     response_format={"type": "json_object"}
# )
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
    original_style_keywords: str = ""
    original_character_keywords: str = ""
    original_world_keywords: str = ""


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
    original_theme: str = ""


# --- 新增：角色数据模型 ---
class Character(BaseModel):
    name: str
    description: str


class CharactersData(BaseModel):
    characters: List[Character]


# -------------------------

# 优化后的图像生成请求模型，只接收项目名
class ImageGenerationRequest(BaseModel):
    task_name: str


class ImageResult(BaseModel):
    final_prompt: str
    url: str

class ImageResponse(BaseModel):
    image_urls: List[str] = []
    image_results: List[ImageResult] = []
    is_local_storage: bool = True


class ProjectData(BaseModel):
    brief: BriefDetails
    story: Story
    images: ImageResponse
    characters_data: CharactersData | None = None  # 新增：项目数据中也保存角色图谱

class ProjectSaveRequest(BaseModel):
    brief: BriefDetails
    story: Story
    characters_data: CharactersData | None = None


class TaskInfo(BaseModel):
    folder_name: str
    title: str
    preview_image_url: str | None = None


# --- FastAPI 应用实例 ---
app = FastAPI()
app.mount("/static/data", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "data")), name="data")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- AI 服务 ---
# step1:创建背景
async def generate_brief_from_ai(keywords: BriefKeywords) -> BriefDetails:
    """使用真实的AI模型根据关键词生成创意简报"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI 客户端未初始化")

    prompt = f"""
        你是一位短视频架构师，擅长将简洁的关键词扩展成适合短视频平台的创意简报。
        你的任务是根据用户提供的三个核心关键词，为一部短片生成一个可指导后续短视频平台视觉创作的“创意简报”。

        特别要求：
        1.  **[美术风格]**：在“style”描述的最前，必须明确提炼出3-5个用于在AI绘画中的“美术风格核心一致性关键词”，并用 "关键词:" 开头。随后，输出对该风格准确的描述，涵盖：色彩与光影、构图与运镜、线条与细节、渲染与质感。
        2.  **[人物角色]**：在“character”描述的最前，必须明确提炼出3-5个用于在AI绘画中的“人物特点核心关键词”，并用 "关键词:" 开头。随后，输出对角色的故事化描述，重点阐述其核心身份、性格、人设等，不用具象的外貌细节，以避免与后续人物图谱生成环节冗余。
        3.  **[核心主题]**：在“world”描述的最前，必须明确提炼出3-5个用于在AI绘画中的“主题核心一致性关键词”，并用 "关键词:" 开头。随后，输出对该世界观的描述，包括其节奏张力、视觉氛围、画面冲击力，要适应短视频风格。

        请严格按照以下JSON格式返回，不要包含任何额外的说明性文本，各50字左右。

        用户输入：
        - 美术风格关键词: {keywords.style} 
        - 角色描述关键词: {keywords.character}  
        - 核心主题关键词: {keywords.world} 

        输出JSON格式：
        {{
          "style": "...",  # 关键词:..., 换行后输出风格细节
          "character": "...",  # 关键词:..., 换行后输出角色
          "world": "..."  # 关键词:..., 换行后输出主题
        }}
        """

    try:
        completion = client.chat.completions.create(
            model=model_for_hard,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={"type": "json_object"}  # 请求JSON输出
        )
        print(f"✅ 创意简报生成成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")

        response_text = completion.choices[0].message.content
        brief_data = json.loads(response_text)
        # 添加原始关键词
        brief_data["original_style_keywords"] = keywords.style
        brief_data["original_character_keywords"] = keywords.character
        brief_data["original_world_keywords"] = keywords.world
        return BriefDetails(**brief_data)
    except (APIStatusError, APIConnectionError) as e:
        raise HTTPException(status_code=500, detail=f"AI 服务调用失败: {str(e)}")
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=500, detail=f"AI 返回格式错误: {str(e)}")


# step2:故事生成
async def generate_story_from_ai(request: StoryRequest) -> AsyncGenerator[str, None]:
    """
    使用“三步走”策略生成故事：1. 生成大纲 -> 2. 生成角色图谱 -> 3. 逐幕扩写与分析
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
            model=model_for_hard,
            messages=[{'role': 'user', 'content': overall_plot_prompt}],
            response_format={"type": "json_object"}
        )
        print(f"✅ 故事大纲生成成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")

        story_structure = json.loads(completion.choices[0].message.content)
        title = story_structure.get("title", f"关于“{request.theme}”的故事")
        plot_outline = story_structure.get("plot_outline", [])

        if len(plot_outline) != 6:
            raise ValueError("AI未能生成包含六幕的剧情大纲。")

        # 将标题和原始主题一起返回
        yield json.dumps({
            "title": title,
            "original_theme": request.theme  # 保存用户输入的原始主题
        }) + "\n"

    except Exception as e:
        print(f"❌ 关键步骤失败：整体故事大纲生成失败: {e}")
        yield json.dumps({"error": f"故事大纲生成失败: {e}"}) + "\n"
        raise StopAsyncIteration

    # --- 第二步：基于大纲，生成角色图谱 ---
    characters_data = {"characters": []}
    try:
        # 先用小模型判断是否有人物 & 核心人物（≥2幕）
        print("--- 正在用小模型筛查核心人物（是否存在人物且出现≥2幕） ---")
        core_detector_prompt = f"""
        你是一名快速检索小助手。请基于六幕剧情大纲判断是否存在“人物”（人类或具有人物属性的拟人角色）。
        目标：只输出在剧情中“至少出现在两幕及以上”的核心人物名单。

        规则：
        1) 把能被复指、推动剧情的主体视为“人物”；道具/物件（如钢笔、山、风、手机）不是人物，除非文本明确以拟人方式作为角色且参与剧情。
        2) 统计每个疑似人物分别出现在第几幕。
        3) 仅保留出现次数≥2的角色，按出现次数降序。
        4) 严格输出JSON，不要解释。

        输入（六幕大纲）：
        {json.dumps(plot_outline, ensure_ascii=False)}

        输出JSON格式：
        {{
          "has_human": true | false,
          "core_characters": [
            {{"name": "角色名", "acts": [1,3,5]}}
          ]
        }}
        """
        detector_completion = client.chat.completions.create(
            model=model_for_easy,
            messages=[{'role': 'user', 'content': core_detector_prompt}],
            response_format={"type": "json_object"}
        )
        detector_result = json.loads(detector_completion.choices[0].message.content)
        has_human = bool(detector_result.get("has_human"))
        core_list = detector_result.get("core_characters") or []
        core_names = [c.get("name") for c in core_list if isinstance(c.get("name"), str) and c.get("name").strip()]

        if not has_human or not core_names:
            print("--- 核心人物为空或不存在人物，跳过角色图谱生成 ---")
            # 明确回传空角色，供前端与后续流程感知
            characters_data = {"characters": []}
            yield json.dumps({
                "characters_data": characters_data,
                "note": "no_core_characters_detected"
            }) + "\n"
        else:
            print(f"--- 识别到核心人物：{core_names} ，开始生成像素级模板 ---")
            # ================= 保留你的原始提示词，仅加“只针对核心人物名单生成”的约束 =================
            character_prompt = f"""
            你是一位世界顶级的角色设定师和概念艺术师。你的任务是为AI绘画模型创建一个“角色身份档案 (Character Identity Profile)”，确保在不同场景下角色形象的绝对一致性。

            核心原则：
            1.  **创建唯一ID**：为每个角色分配一个独特且具体的代号，作为跨画面的“身份识别码”。
            2.  **定义强制锚点**：明确指出哪些视觉特征是“不可变的”。
            3.  **面部视觉档案**：将所有构成“同一个人”的脸部细节，封装为一个高优先级的视觉档案，供AI在需要时调用。

            描述内容需包含：
            -   [核心特征]：一句话总结角色的核心身份、职业或独特风格，例如：“一个冷酷的赛博朋克特工”。
            -   [一致性指令]：**这是最重要的部分，也是最高优先级的视觉锚点。** 用强制性、命令式的语言，列出3-4个在任何情况下都不能改变的核心视觉锚点，特别是与脸部无关的特征。
                **提示**：请包含身体印记（疤痕、纹身）、独特的身体改造（义肢）、或标志性的配饰。
                例如：“(必须始终拥有: 一个发光的赛博义肢左臂, 一枚颈部条形码纹身)”。
            -   [面部视觉档案]：**请提供以下脸部细节的完整描述，这是构成该角色灵魂的核心，必须被严格遵循。**
                -   **脸型与轮廓**：请具体描述脸型、下巴、颧骨和下颌线，例如：“脸型瘦削，下颌线硬朗，带有战斗痕迹的细微疤痕”。
                -   **皮肤质感与瑕疵**：请描述肤色的具体色调、皮肤的纹理和任何可见的瑕疵，例如：“肤色古铜色，皮肤质地粗糙，带有细小毛孔与汗珠”。
                -   **眼睛**：请描述眼睛的颜色、形状、瞳孔细节和眼神气质，例如：“眼睛为金黄色，瞳孔呈竖线状，虹膜带有微弱火光，目光锐利且充满挑衅”。
                -   **眉毛**：请描述眉毛的形状、浓密程度和走向，例如：“眉毛粗浓，呈倒八字形，向上挑起”。
                -   **鼻子**：请描述鼻子的形态和鼻梁的形状，例如：“鼻子中等大小，鼻梁直”。
                -   **嘴唇**：请描述嘴唇的厚度和轮廓，以及嘴角的倾向，例如：“嘴唇中等厚度，嘴角常带冷笑，唇线分明”。
            -   [身材与体态]：请描述角色的身高、体型、肌肉轮廓以及独特的站姿，例如：“身高约175厘米，体型精瘦但肌肉紧实，站姿略带前倾，随时准备攻击”。
            -   [服装与配饰]：请描述服装的风格、材质、颜色和任何标志性的配饰，例如：“身穿红色混金丝战袍，材质为丝绸，表面有微弱反光，头顶佩戴金色凤翅紫金冠，冠上有三根长翎随风飘动”。

            【只针对下列核心人物生成模板，不得新增或删除人物；按给定顺序输出】
            核心人物名单：{json.dumps(core_names, ensure_ascii=False)}

            [剧情大纲]
            {json.dumps(plot_outline, ensure_ascii=False)}
            [用户参考角色]
            {request.brief.character}
            [输出要求]
            -   严格输出JSON，不要包含任何额外说明。
            -   仅输出“核心人物名单”中的角色，每个角色一条。

            [输出格式警告]
            -   请绝对确保你的输出是一个完整的、格式正确的、有效的JSON对象。
            -   **所有键名和字符串值都必须用双引号包围。**
            -   不要在JSON对象之外添加任何额外文字。

            [输出JSON格式示例]
            {{
                "characters": [
                    {{
                        "name": "角色A",
                        "ai_prompt_id": "char_cyberpunk_agent_kai",
                        "description": "[角色ID]：char_cyberpunk_agent_kai。[一致性指令]：必须始终拥有: 一个发光的赛博义肢左臂, 一枚颈部条形码纹身。[面部视觉档案]：脸型瘦削，下巴有旧伤疤，棱角分明。眼睛是深蓝色电子义眼，瞳孔会发出微弱蓝光，眼角有细微电路纹理。嘴唇紧抿，表情冷峻。[核心特征]：一个冷酷的赛博朋克特工。[身材与体态]：身材精干，肌肉线条明显，总是保持警惕的站姿。[服装与配饰]：黑色战术背心，深灰色高科技长裤。"
                    }}
                ]
            }}
            """
            char_completion = client.chat.completions.create(
                model=model_for_hard,
                messages=[{'role': 'user', 'content': character_prompt}],
                response_format={"type": "json_object"}
            )
            print(f"✅ 角色图谱生成成功 - Token 消耗: {char_completion.usage.total_tokens}")
            characters_data = json.loads(char_completion.choices[0].message.content)

            # 使用 Pydantic 模型验证
            CharactersData(**characters_data)
            yield json.dumps({"characters_data": characters_data}) + "\n"

    except Exception as e:
        print(f"❌ 角色图谱生成失败: {e}")
        yield json.dumps({"error": f"角色图谱生成失败: {e}"}) + "\n"

    # --- 第三步：基于大纲，循环生成每一幕的详细信息 ---
    previous_actions = []
    for i in range(1, 7):
        current_plot_point = plot_outline[i - 1]

        segment_data = {}
        try:
            # --- 步骤 3a: 创意内容生成 ---
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
                model=model_for_hard, messages=[{'role': 'user', 'content': creative_prompt}],
                response_format={"type": "json_object"}
            )
            creative_data = json.loads(creative_completion.choices[0].message.content)
            segment_data.update(creative_data)
            print(f"✅ 第 {i} 幕创意内容生成成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")

            # --- 步骤 3b: 基于 action 内容进行数据分析 ---
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
                    "喜悦": 0.0, "悲伤": 0.0, "愤怒": 0.0,
                    "恐惧": 0.0, "惊讶": 0.0, "紧张": 0.0
                }}
            }}
            """
            analytics_completion = client.chat.completions.create(
                model=model_for_easy, messages=[{'role': 'user', 'content': analytics_prompt}],
                response_format={"type": "json_object"}
            )
            analytics_data = json.loads(analytics_completion.choices[0].message.content)
            segment_data['analytics'] = analytics_data
            print(f"✅ 第 {i} 幕分析指数计算成功。")
            print(f"✅ 第 {i} 幕分析指数计算成功 - Token 消耗: {analytics_completion.usage.total_tokens} (输入: {analytics_completion.usage.prompt_tokens}, 输出: {analytics_completion.usage.completion_tokens})")

            # --- 合并、验证并输出 ---
            Segment(**segment_data)
            previous_actions.append(segment_data['action'])
            yield json.dumps(segment_data) + "\n"

        except (ValidationError, json.JSONDecodeError, KeyError) as e:
            print(f"❌ 第 {i} 幕生成过程中失败: {e}")
            yield json.dumps({"error": f"第 {i} 幕生成失败: {e}"}) + "\n"
            continue
        except Exception as e:
            yield json.dumps({"error": f"AI服务调用失败: {e}"}) + "\n"
            raise StopAsyncIteration


# step3.1基于前端传回的背景信息和提示词生成AI绘画的提示词
async def refine_prompt_with_ai(brief: BriefDetails, segment_prompt: str, characters_data: CharactersData) -> str:
    """
    使用“角色图谱注入法”，将固定的角色设定与单幕Prompt融合成高度优化的最终Prompt。
    """
    if not client:
        return f"{segment_prompt}, {brief.style}, {brief.character}"

    # 将角色图谱数据格式化为字符串，以便注入到Prompt中
    character_templates = "\n".join(
        [f"- **{char.name}**: {char.description}" for char in characters_data.characters]
    )
    if not character_templates:
        character_templates = "无可用角色模板。"

    # --- 核心修改：在字符串前加上 'f'，使其成为 f-string ---
    refiner_prompt = f"""
    你是一位世界顶级的AI绘画提示词（Prompt）架构师，同时也是一位精通各类艺术风格的视觉专家。
    你的任务是基于一个“全局设定”、“角色图谱”和一个“特定场景描述”，构建一个单一、完整、且高度优化的最终提示词。

    核心原则：
    1.  **[Style]**模块是所有画面的“系列级绝对锚点”，定义了宏观的视觉语言。
    2.  **[Character]**模块是人物的“人物级绝对锚点”，它将包含**AI专用的ID和强制性一致性指令**，并根据当前场景的可见性进行智能筛选。
    3.  **你的任务是确保AI在[Style]设定的框架内，精确地呈现[Character]中与场景相关的关键细节。**
    4.  输出必须是一个单一、完整的字符串，内部使用特定的结构化分隔符。

    ---
    [全局设定 (GLOBAL BRIEF)]
    # 美术风格: {brief.style}
    # 核心主题: {brief.world}
    ---
    [角色图谱 (CHARACTER TEMPLATES) - 这是人物级绝对锚点]
    {character_templates}
    ---
    [特定场景描述 (SPECIFIC SCENE PROMPT)]
    {segment_prompt}
    ---
    [你的核心任务 (MANDATORY EXECUTION STEPS)]
    1.  **风格提炼与注入**: 根据[全局设定]中的`美术风格`和`核心主题`，提炼并生成一个包含色彩、质感、构图、光影等核心元素的精炼风格提示词。此全量风格提示词必须作为 **[Style]** 模块注入到最终提示词的最前端。
    2.  **场景解析**: 仔细阅读“特定场景描述”，提取核心动作、环境、情绪和**镜头视角（如：特写、全景、俯瞰）**，将其转化为精准的视觉语言，作为 **[Scene]** 模块。
    3.  **角色识别与智能筛选**: 判断“特定场景描述”中是否提到了“角色图谱”里的任何角色。如果识别到，根据**场景的镜头视角**，智能地筛选并提取**仅与当前画面可见性相关**的角色细节。
        -   **特别注意：** 无论镜头远近，都**必须**将该角色的**[角色ID]**和**[一致性指令]**完整地注入，这是最高权重的锚点。
        -   **若为远景或全景：** 在ID和指令后，提取`[服装与配饰]`、`[身材与体态]`等远景可见的特征。
        -   **若为中景或特写：** 在ID和指令后，重点提取`[面部与表情]`、`[服装与配饰]`和`[身材与体态]`。
        -   **若角色未出现或不重要：** 该模块可省略。
        将精选后的描述组合成 **[Character]** 模块。
    4.  **整合输出**: 将解析后的风格、场景和精选的角色描述，按照以下格式组合成一个流畅、详细、结构化的单一字符串。

    [输出格式]
    **[Style]**<自动生成的精炼风格提示词>, **[Scene]**<场景描述>, **[Character]**<精选的角色描述>, **[Keywords]**<情绪与核心概念关键词>
    -   每个部分都由逗号连接，不同部分之间使用粗体标签（如 **[Style]**）作为清晰的区块分隔符。

    [输出要求]
    -   绝对禁止任何形式的解释、标题或前言。
    -   必须使用中文进行输出。
    -   最终结果必须是一个**单一的、结构化的字符串**，严格遵循上述[输出格式]，且[Style]部分永远位于最前端。
    """

    try:
        completion = client.chat.completions.create(
            model=model_for_hard,
            messages=[{'role': 'user', 'content': refiner_prompt}]
        )
        print(
            f"    - 提示词精炼成功 - Token 消耗: {completion.usage.total_tokens} (输入: {completion.usage.prompt_tokens}, 输出: {completion.usage.completion_tokens})")
        refined_prompt = completion.choices[0].message.content.strip()
        return refined_prompt.replace('\n', ', ').replace('\r', '')
    except Exception as e:
        print(f"    - 提示词精炼失败，将使用原始提示词。错误: {e}")
        style_keywords = brief.style.split('关键词:')[1].split('\n')[0].strip() if '关键词:' in brief.style else ''
        char_keywords = brief.character.split('关键词:')[1].split('\n')[
            0].strip() if '关键词:' in brief.character else ''
        return f"{segment_prompt}, {char_keywords}, {style_keywords}"


# step3.2基于合成的提示词生成图像
async def generate_images_from_ai(request: ImageGenerationRequest) -> ImageResponse:
    semaphore = asyncio.Semaphore(2)
    tasks = []

    # 从项目文件中加载所有数据
    project_file_path = os.path.join(os.path.dirname(__file__), "..", "data", request.task_name, "project_data.json")
    if not os.path.exists(project_file_path):
        raise HTTPException(status_code=404, detail=f"任务 '{request.task_name}' 的项目文件未找到。")
    try:
        with open(project_file_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
            # 使用 Pydantic 模型加载和验证数据
            loaded_project = ProjectData(**project_data)
            brief = loaded_project.brief
            segments = loaded_project.story.segments
            characters_data = loaded_project.characters_data or CharactersData(characters=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载项目 '{request.task_name}' 数据失败: {str(e)}")

    async def controlled_generation(segment):
        async with semaphore:
            try:
                # --- 核心修改：从加载的数据中获取 brief 和 characters_data ---
                final_prompt = await refine_prompt_with_ai(brief, segment.firstFramePrompt, characters_data)
                print(f"    - 最终生成提示词 (ID: {segment.id}): {final_prompt}")

                safe_title = re.sub(r'[\\/*?:"<>|]', "", segment.title)
                file_name = f"{segment.id:02d}_{safe_title}.png"
                file_path = os.path.join(os.path.dirname(project_file_path), file_name)

                url = await asyncio.to_thread(call_wanx_sync_and_save, final_prompt, QWEN_API_KEY, file_path)

                if "placehold.co" in url or "Error" in url:
                    raise Exception(f"图片生成或下载失败 (ID: {segment.id})")

                await asyncio.sleep(1)
                return {"final_prompt": final_prompt, "url": url}
            except Exception as e:
                print(f"❌ 图片生成任务失败 (ID: {segment.id}): {e}")
                return e

    for segment in segments:
        tasks.append(controlled_generation(segment))

    print(f"正在为 {len(tasks)} 个段落生成图片 (并发限制: 2)...")
    all_results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"✅ 所有图片生成任务已执行完毕。")

    successful_results = []
    failed_tasks_count = 0
    for result in all_results_or_errors:
        if isinstance(result, Exception):
            failed_tasks_count += 1
        else:
            successful_results.append(result)

    if failed_tasks_count > 0:
        print(f"⚠️  {failed_tasks_count} 个图片生成任务失败。")
    if not successful_results:
        raise HTTPException(status_code=500, detail="所有图片生成任务均失败，请检查后端日志。")

    try:
        # 更新项目文件中的图片信息
        image_urls = [result['url'] for result in successful_results]
        loaded_project.images.image_urls = image_urls
        
        # 构建更完整的图像结果数据
        project_dict = loaded_project.dict(by_alias=True)
        # 添加 image_results 字段
        project_dict['images']['image_results'] = [
            {"final_prompt": result['final_prompt'], "url": result['url']} 
            for result in successful_results
        ]
        # 确保 is_local_storage 字段存在
        project_dict['images']['is_local_storage'] = True
        
        with open(project_file_path, 'w', encoding='utf-8') as f:
            json.dump(project_dict, f, ensure_ascii=False, indent=4)
        print(f"✅ 项目数据已更新并保存到: {project_file_path}")

        return ImageResponse(image_urls=image_urls)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像生成后，项目保存过程中发生错误: {str(e)}")


# step3.3保存生成图像到本地
def get_unique_save_directory(base_name: str) -> str:
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, "..", "data")
    safe_base_name = re.sub(r'[\\/*?:"<>|]', "", base_name).strip() or "Untitled_Task"
    task_dir = os.path.join(data_dir, safe_base_name)
    counter = 1
    while os.path.exists(task_dir):
        task_dir = os.path.join(data_dir, f"{safe_base_name}-{counter}")
        counter += 1
    os.makedirs(task_dir, exist_ok=True)
    print(f"创建任务文件夹: {task_dir}")
    return task_dir


# 工具函数：使用qwenapi生成图像
def call_wanx_sync_and_save(prompt: str, api_key: str, save_path: str) -> str:
    print(f"  - 正在为 '{os.path.basename(save_path)}' 生成图片...")
    rsp = ImageSynthesis.call(
        api_key=api_key, model=model_for_image, prompt=prompt, n=1,
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


# --- API 接口 ---
@app.post("/generate_brief", response_model=BriefDetails)
async def generate_brief(keywords: BriefKeywords):
    return await generate_brief_from_ai(keywords)


@app.post("/generate_story_details")
async def generate_story_details(request: StoryRequest):
    return StreamingResponse(generate_story_from_ai(request), media_type="application/x-ndjson")


@app.post("/generate_images", response_model=ImageResponse)
async def generate_images_endpoint(request: ImageGenerationRequest):
    return await generate_images_from_ai(request)

@app.post("/save_project")
async def save_project(request: ProjectSaveRequest):
    try:
        # 使用故事标题创建唯一的任务文件夹名
        task_name = get_unique_save_directory(request.story.title)
        project_file_path = os.path.join(task_name, "project_data.json")

        # 获取基础数据
        brief_data = request.brief.dict()
        story_data = request.story.dict()

        # 确保 original_theme 字段存在，并移除 original_outline
        if "original_theme" not in story_data and hasattr(request.story, "original_theme"):
            story_data["original_theme"] = request.story.original_theme
        if "original_outline" in story_data:
            del story_data["original_outline"]

        # 确保 original_*_keywords 字段存在
        for key in ["original_style_keywords", "original_character_keywords", "original_world_keywords"]:
            if key not in brief_data and hasattr(request.brief, key):
                brief_data[key] = getattr(request.brief, key)

        # 将所有数据打包并保存到文件
        project_data = {
            "brief": brief_data,
            "story": story_data,
            "images": {"image_urls": [], "image_results": [], "is_local_storage": True}, # 图像信息初始为空
            "characters_data": request.characters_data.dict() if request.characters_data else {'characters': []}
        }
        with open(project_file_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=4)

        return {"task_name": os.path.basename(task_name)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存项目失败: {str(e)}")

# --- 项目管理接口 ---
@app.get("/tasks", response_model=List[TaskInfo])
async def list_tasks():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.exists(data_dir):
        return []

    tasks = []
    # os.listdir() 返回的是无序的列表
    # os.path.getmtime() 获取文件的最后修改时间

    # 1. 获取所有文件夹，并按修改时间倒序排序
    all_folders = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ]
    all_folders.sort(key=os.path.getmtime, reverse=True)

    # 2. 遍历排序后的前7个文件夹
    for folder_path in all_folders[:7]:
        folder_name = os.path.basename(folder_path)
        project_file = os.path.join(folder_path, "project_data.json")

        # 确保项目文件存在
        if os.path.exists(project_file):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    title = data.get("story", {}).get("title", folder_name)

                    # 从项目文件中读取预览图 URL
                    preview_image_url = None
                    if 'image_urls' in data.get('images', {}):
                        image_urls = data['images']['image_urls']
                        if image_urls:
                            preview_image_url = image_urls[0]

                    tasks.append(
                        TaskInfo(folder_name=folder_name, title=title, preview_image_url=preview_image_url)
                    )
            except Exception as e:
                print(f"读取项目失败 {folder_name}: {e}")

    return tasks

@app.get("/task/{task_name}", response_model=ProjectData)
async def load_task(task_name: str):
    project_file = os.path.join(os.path.dirname(__file__), "..", "data", task_name, "project_data.json")
    if not os.path.exists(project_file):
        raise HTTPException(status_code=404, detail="项目文件未找到")
    try:
        with open(project_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # --- 数据兼容性处理 ---
            if 'images' in data and 'image_results' in data['images']:
                new_urls = []
                for item in data['images'].get('image_results', []):
                    if isinstance(item, dict) and 'url' in item:
                        new_urls.append(item['url'])
                data['images']['image_urls'] = new_urls

            # 为旧项目文件补充空的 characters_data 字段，避免校验失败
            if 'characters_data' not in data:
                data['characters_data'] = {'characters': []}

            return ProjectData(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载项目数据失败: {e}")


@app.get("/", response_class=FileResponse)
async def read_index():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if not os.path.exists(frontend_path):
        return {"error": "index.html not found"}
    return FileResponse(frontend_path)

# 添加主入口点，适用于本地运行和 Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)