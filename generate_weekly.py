import os
import sys
from datetime import datetime, timedelta
from openai import OpenAI
from tavily import TavilyClient

# --- Config ---
OUTPUT_DIR = "AI与机器人部"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# --- Init clients ---
deepseek = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# --- Date range ---
today = datetime.now()
week_ago = today - timedelta(days=7)
date_str = today.strftime("%Y-%m-%d")
range_cn = f"{week_ago.strftime('%Y年%m月%d日')} — {today.strftime('%Y年%m月%d日')}"
month_en = today.strftime("%B %Y")
month_cn = f"{today.year}年{today.month}月"

# --- Search sections (name -> list of search queries) ---
SECTIONS = {
    "AI 数学与推理突破": [
        f"AI artificial intelligence math reasoning breakthrough research {month_en}",
        f"大模型 数学推理 突破 最新进展 {month_cn}",
    ],
    "AI for Science（科学智能）": [
        f"AI for Science breakthrough research paper {month_en}",
        f"AI4Science 人工智能 科学研究 突破 {month_cn}",
    ],
    "具身智能与机器人突破": [
        f"embodied AI humanoid robot breakthrough research {month_en}",
        f"具身智能 人形机器人 突破 进展 {month_cn}",
    ],
    "机械臂本体（结构·驱动·运动学·控制）": [
        f"robotic arm manipulator mechanism design kinematics dynamics control research paper {month_en}",
        f"机械臂 本体 结构设计 控制 绳驱 软体 连续体 研究 {month_cn}",
    ],
    "DeepSeek 专栏": [
        f"DeepSeek AI latest news research {month_en}",
        f"DeepSeek 深度求索 最新进展 {month_cn}",
    ],
    "顶级会议与学术动态": [
        f"AI academic paper top conference ICML NeurIPS CVPR {today.year} latest",
        f"人工智能 顶级会议 论文 趋势 {month_cn}",
    ],
    "AI 基础设施与能效": [
        f"AI infrastructure energy efficiency breakthrough {month_en}",
        f"AI 算力 能效 芯片 突破 {month_cn}",
    ],
    "产业与政策": [
        f"AI robotics industry policy investment funding {month_en}",
        f"人工智能 机器人 产业政策 投融资 {month_cn}",
    ],
}


def search_all() -> dict[str, list[dict]]:
    """Run Tavily searches for all sections. Returns {section_name: [results]}."""
    all_results: dict[str, list[dict]] = {}
    total_queries = sum(len(qs) for qs in SECTIONS.values())
    done = 0

    for section_name, queries in SECTIONS.items():
        section_results: list[dict] = []
        for q in queries:
            done += 1
            print(f"[{done}/{total_queries}] 搜索: {section_name} ...", flush=True)
            try:
                resp = tavily.search(
                    query=q,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=False,
                )
                section_results.extend(resp.get("results", []))
            except Exception as e:
                print(f"  !! 搜索失败: {e}", flush=True)

        # Deduplicate by URL
        seen = set()
        unique = []
        for r in section_results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                unique.append(r)
        all_results[section_name] = unique
        print(f"  -> {len(unique)} 条去重结果", flush=True)

    return all_results


def format_results(all_results: dict[str, list[dict]]) -> str:
    """Convert search results to a text blob for the LLM prompt."""
    parts = []
    for section_name, results in all_results.items():
        parts.append(f"\n### {section_name} ({len(results)} 条)")
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            content = r.get("content", "")
            parts.append(f"\n[{i}] {title}\n    URL: {url}\n    摘要: {content}")
    return "\n".join(parts)


def build_user_prompt(results_text: str) -> str:
    return f"""请根据以下搜索到的新闻素材，生成一份完整的 HTML 周报文件。

## 要求

### 时间范围
仅收录 **{range_cn}** 这 7 天内的新闻。超过 7 天的、明显过时的新闻请筛掉。

### 版面结构（8 个板块，必须全部出现）
1. 本期焦点（一段话概括本周最重要进展）
2. AI 数学与推理突破
3. AI for Science（科学智能）
4. 具身智能与机器人突破
5. 机械臂本体（结构·驱动·运动学·控制）
6. DeepSeek 专栏
7. 顶级会议与学术动态
8. AI 基础设施与能效
9. 产业与政策

### 内容要求
- 每条新闻要有：标题、日期/发表期刊、内容摘要（中文）、意义点评、来源链接
- 中英文来源均翻译为中文呈现，保留原始来源链接
- 偏重学术论文和技术突破，产业与政策为辅
- 最终输出 **纯 HTML**，无需 markdown 包裹，直接从 <!DOCTYPE html> 开始
- **严禁使用 JavaScript**：所有内容必须是静态 HTML，直接写在卡片元素里，禁止使用 <script> 标签动态注入内容，禁止使用 innerHTML 或 getElementById 等方式填充数据

### 样式要求
- 白底主题（--bg: #ffffff），淡灰卡片，深色文字
- 深蓝色强调色（--accent: #2563eb）
- 板块标题带条目计数
- 每条新闻一张卡片，含标签、来源链接
- 顶部有统计栏（总条目数、板块数、引用信源数）
- 底部有页脚（生成时间、免责声明）
- 字体用系统默认中文字体栈

### 搜索素材
{results_text}"""


SYSTEM_PROMPT = """你是一个专业的 AI 与机器人领域学术新闻编辑。你的任务是根据搜索素材生成完整的 HTML 周报。

核心原则：
1. 只保留 7 天内的新闻（即使用户素材中混入了更早的，也要筛掉）
2. 优先学术论文和技术突破，产业政策为辅
3. 所有内容翻译为简体中文，保留原始 URL
4. 严格输出纯 HTML（从 <!DOCTYPE html> 开始），不要 markdown 代码块包裹

HTML 要求：
- 内嵌 <style>，白底主题
- 每个板块一个 <div class="section">
- 每条新闻一个 <div class="card">，包含标题、日期、摘要、标签、来源链接
- 顶部统计栏：总条目数、板块数
- 如果某个板块确实没有本周相关新闻，标注"本周暂无相关重大进展"

CSS 变量参考（白底主题）：
--bg: #ffffff; --card-bg: #f8f9fb; --text: #1a1a2e; --text-secondary: #555;
--accent: #2563eb; --border: #e2e5ea; 标签用浅色底+对应色字"""


def generate_html(results_text: str) -> str:
    """Call DeepSeek to generate the HTML report."""
    print("调用 DeepSeek API 生成周报 HTML ...", flush=True)
    resp = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(results_text)},
        ],
        temperature=0.3,
        max_tokens=16384,
    )
    content = resp.choices[0].message.content or ""

    # Strip any leading text before <!DOCTYPE or <html
    doctype_pos = content.find("<!DOCTYPE")
    html_pos = content.find("<html")
    if doctype_pos > 0:
        content = content[doctype_pos:]
    elif html_pos > 0:
        content = content[html_pos:]

    # Strip markdown code fences
    lines = content.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    content = "\n".join(lines)

    return content


def main():
    print(f"=== AI 与机器人周报生成 ===")
    print(f"日期范围: {range_cn}")
    print(f"输出文件: {OUTPUT_DIR}/{date_str}.html")
    print()

    # Step 1: Search
    print("--- 第一步：Tavily 搜索 ---")
    all_results = search_all()
    total = sum(len(v) for v in all_results.values())
    print(f"共获取 {total} 条搜索结果\n")

    # Step 2: Format for LLM
    results_text = format_results(all_results)

    # Step 3: Generate HTML
    print("--- 第二步：DeepSeek 生成 HTML ---")
    html = generate_html(results_text)
    print(f"HTML 长度: {len(html)} 字符\n")

    # Step 4: Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{date_str}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"--- 完成！周报已保存至 {output_path} ---")


if __name__ == "__main__":
    main()
