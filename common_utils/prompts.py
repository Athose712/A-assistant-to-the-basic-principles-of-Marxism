"""
This file stores all prompt templates for the question generation agent.
Centralizing prompts here makes them easier to manage, version, and adapt
for different agents (e.g., Mao's Thoughts, Xi's Thoughts).
"""
from langchain_core.prompts import PromptTemplate

# --- Generic Prompt for Single Question Type ---
# This template is used when the user requests a single type of question.
# It's dynamically formatted with the specific type's requirements.
SINGLE_TYPE_PROMPT_TEMPLATE = PromptTemplate.from_template("""
你是一位资深的{subject_name}课程教师，具有丰富的出题经验。请根据以下要求生成高质量的题目。

**任务要求：**
- 主题：{topic}
- 题目数量：{num_questions}道
- 难度等级：{difficulty}
- 题目类型：{question_type_specific_name}

**参考资料：**
{context}

**出题要求：**
1. 题目必须严格基于提供的参考资料内容。
2. {difficulty}难度的题目特点：
   - 简单：考查基本概念和定义的理解。
   - 中等：考查概念间的关系和应用。
   - 困难：考查深层理解、分析和综合运用能力。
3. {format_requirements}
4. 语言表达要准确、严谨。

**原始用户需求：** 
{user_input}

**输出格式：**
{output_format_example}

请严格按照上述需求生成题目。
""")

# --- Specific Formatting Requirements for Each Question Type ---
# These dictionaries define the unique parts of the prompt for each question type.
# This avoids repeating the main template.
QUESTION_TYPE_CONFIG = {
    "选择题": {
        "question_type_specific_name": "选择题",
        "format_requirements": "每道选择题包含：题干、4个选项（A、B、C、D）、正确答案和简要解析。\n4. 选项设计要合理，干扰项要有一定迷惑性。\n5. 为便于后续按需展示，请确保每个要素各自独立成行，并以如下关键词开头：‘题干：’、‘A.’、‘B.’、‘C.’、‘D.’、‘正确答案：’、‘解析：’。",
        "output_format_example": """题目1：
题干：[具体题目内容]
A. [选项A]
B. [选项B]
C. [选项C]
D. [选项D]
正确答案：[正确选项]
解析：[简要解析说明]

题目2：
..."""
    },
    "判断题": {
        "question_type_specific_name": "判断题",
        "format_requirements": "判断题格式：题干 + 正确答案（正确/错误）+ 简要解析。\n为便于后续按需展示，请确保每个要素各自独立成行，并以如下关键词开头：‘题干：’、‘正确答案：’、‘解析：’。",
        "output_format_example": """题目1：
题干：[具体题目内容]
正确答案：[正确/错误]
解析：[简要解析说明]

题目2：
..."""
    },
    "简答题": {
        "question_type_specific_name": "材料分析/简答题",
        "format_requirements": "每道材料分析/简答题包含：题干（可提供材料或问题描述）、参考答案、简要解析。\n为便于后续按需展示，请确保每个要素各自独立成行，并以如下关键词开头：‘题干：’、‘参考答案：’、‘解析：’。",
        "output_format_example": """题目1：
题干：[具体题目内容]
参考答案：[答案内容]
解析：[简要解析说明]

题目2：
..."""
    }
}

# --- Prompt for Mixed Question Types ---
# This template is used when the user requests multiple types of questions in one go.
MIXED_TYPE_PROMPT_TEMPLATE = PromptTemplate.from_template("""
你是一位资深的{subject_name}课程教师，具有丰富的出题经验。请根据以下要求生成高质量的题目。

**任务要求：**
- 主题：{topic}
- 题目类型及数量：
{type_details}
- 难度等级：{difficulty}

**参考资料：**
{context}

**出题要求：**
1. 题目必须严格基于提供的参考资料内容。
2. {difficulty}难度的题目特点：
   - 简单：考查基本概念和定义的理解。
   - 中等：考查概念间的关系和应用。
   - 困难：考查深层理解、分析和综合运用能力。
3. 各题型格式要求：
   - 选择题：题干 + 4个选项（A、B、C、D）+ 正确答案 + 简要解析。
   - 判断题：题干 + 正确答案（正确/错误）+ 简要解析。
    - 材料分析/简答题：题干（可含材料）+ 参考答案 + 简要解析。
   为便于系统在首次展示时隐藏答案与解析，请确保各要素各自独立成行，并以如下关键词开头：
   ‘题干：’、‘A.’、‘B.’、‘C.’、‘D.’、‘正确答案：’、‘参考答案：’、‘解析：’。
4. 语言表达要准确、严谨。

**原始用户需求：** 
{user_input}

**输出格式示例：**
选择题1：
题干：[具体题目内容]
A. [选项A]
B. [选项B]
C. [选项C]
D. [选项D]
正确答案：[正确选项]
解析：[简要解析说明]

判断题1：
题干：[具体题目内容]
正确答案：[正确/错误]
解析：[简要解析说明]

简答题1：
题干：[具体题目内容]
参考答案：[答案内容]
解析：[简要解析说明]

请按照题型分类并保持题号连续。
""")

# --- Addendum for "Hard" Difficulty ---
# This text is appended to the main prompt when the user requests "困难" (hard) questions
# to guide the LLM in generating more challenging content.
DIFFICULTY_ADDENDUM_HARD = """
**困难题目额外要求：**
- 题干应包含复杂情境或案例，需要学生综合运用原理进行深度分析与评价。
- 对相关概念、范畴或理论命题进行比较、辨析或批判，突出辩证思维。
- 选择题：干扰项需具备高度迷惑性，与正确答案存在关键差异但概念接近。
- 判断题：可设置常见谬误或易混淆表述，引导学生进行严谨辨析。
- 简答/材料分析题：要求多角度论证，联系现实并提出评价与思考。
- 解析部分需展示推理链条或思考步骤，而不仅给出结论。
""" 