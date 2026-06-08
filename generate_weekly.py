"""
AI 与机器人周报生成器

通过 Tavily API 搜索最新 AI/机器人学术新闻，
使用 DeepSeek API 生成完整 HTML 周报，
并自动维护静态化 index.html 首页。

在 GitHub Actions 中每周一 UTC 01:00 定时运行。
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from openai import OpenAI
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
OUTPUT_DIR = "AI与机器人部"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# 安全阈值
MIN_SEARCH_RESULTS = 8       # 全局搜索结果最少条数
MIN_HTML_BYTES = 1500        # 极低底线：仅用于拦截 API 完全返回空/错误的情况
                              # 合法周报即使全是"本周暂无"也远超此值（CSS+结构≈5KB）

# ---------------------------------------------------------------------------
# 客户端初始化
# ---------------------------------------------------------------------------

def _init_clients():
    """初始化外部 API 客户端，校验必需环境变量。"""
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")

    missing = []
    if not deepseek_key:
        missing.append("DEEPSEEK_API_KEY")
    if not tavily_key:
        missing.append("TAVILY_API_KEY")
    if missing:
        print(f"错误: 缺少必需的环境变量: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    deepseek = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )
    tavily = TavilyClient(api_key=tavily_key)
    return deepseek, tavily


# ---------------------------------------------------------------------------
# 日期范围（UTC，与 GitHub Actions cron 对齐）
# ---------------------------------------------------------------------------

def _compute_date_range():
    """计算本周报告的日期范围，统一使用 UTC。"""
    today = datetime.now(timezone.utc)
    week_ago = today - timedelta(days=7)

    return {
        "today": today,
        "week_ago": week_ago,
        "iso_date": today.strftime("%Y-%m-%d"),          # 文件名用
        "range_cn": (
            f"{week_ago.strftime('%Y年%m月%d日')}"
            f" — {today.strftime('%Y年%m月%d日')}"
        ),
        "from_date": week_ago.strftime("%Y-%m-%d"),      # 搜索用
        "to_date": today.strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# 搜索板块定义
# ---------------------------------------------------------------------------

def _build_sections(dr: dict) -> dict[str, list[str]]:
    """构建各板块的 Tavily 搜索词（含精确日期范围）。"""
    frm = dr["from_date"]
    to = dr["to_date"]

    return {
        "AI 数学与推理突破": [
            f"AI artificial intelligence math reasoning breakthrough research {frm} {to}",
            f"大模型 数学推理 突破 最新进展 {frm}",
        ],
        "AI for Science（科学智能）": [
            f"AI for Science breakthrough research paper {frm} {to}",
            f"AI4Science 人工智能 科学研究 突破 {frm}",
        ],
        "具身智能与机器人突破": [
            f"embodied AI humanoid robot breakthrough research {frm} {to}",
            f"具身智能 人形机器人 突破 进展 {frm}",
        ],
        "机械臂本体（结构·驱动·运动学·控制）": [
            f"robotic arm manipulator mechanism design kinematics dynamics control research paper {frm} {to}",
            f"机械臂 本体 结构设计 控制 绳驱 软体 连续体 研究 {frm}",
        ],
        "DeepSeek 专栏": [
            f"DeepSeek AI latest news research {frm} {to}",
            f"DeepSeek 深度求索 最新进展 {frm}",
        ],
        "顶级会议与学术动态": [
            f"AI academic paper top conference ICML NeurIPS CVPR latest {frm} {to}",
            f"人工智能 顶级会议 论文 趋势 {frm}",
        ],
        "AI 基础设施与能效": [
            f"AI infrastructure energy efficiency breakthrough {frm} {to}",
            f"AI 算力 能效 芯片 突破 {frm}",
        ],
        "产业与政策": [
            f"AI robotics industry policy investment funding {frm} {to}",
            f"人工智能 机器人 产业政策 投融资 {frm}",
        ],
    }


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------

def search_all(sections: dict[str, list[str]], tavily: TavilyClient) -> dict[str, list[dict]]:
    """对每个板块执行 Tavily 搜索，去重后返回。"""
    all_results: dict[str, list[dict]] = {}
    total_queries = sum(len(qs) for qs in sections.values())
    done = 0

    for section_name, queries in sections.items():
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

        # URL 去重
        seen: set[str] = set()
        unique: list[dict] = []
        for r in section_results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                unique.append(r)
        all_results[section_name] = unique
        print(f"  -> {len(unique)} 条去重结果", flush=True)

    return all_results


# ---------------------------------------------------------------------------
# 结果格式化 & Prompt 构造
# ---------------------------------------------------------------------------

def format_results(all_results: dict[str, list[dict]]) -> str:
    """将搜索结果拼成文本块供 LLM 使用。"""
    parts: list[str] = []
    for section_name, results in all_results.items():
        parts.append(f"\n### {section_name} ({len(results)} 条)")
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            content = r.get("content", "")
            parts.append(
                f"\n[{i}] {title}\n"
                f"    URL: {url}\n"
                f"    摘要: {content}"
            )
    return "\n".join(parts)


def build_user_prompt(results_text: str, dr: dict) -> str:
    """构造发送给 DeepSeek 的完整 User Prompt。"""
    return f"""请根据以下搜索到的新闻素材，生成一份完整的 HTML 周报文件。

## 要求

### 时间范围
仅收录 **{dr['range_cn']}** 这 7 天内的新闻。超过 7 天的、明显过时的新闻请筛掉。

### 版面结构（9 个板块，必须全部出现）
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
- **严禁使用 JavaScript**：所有内容必须是静态 HTML，直接写在卡片元素里，
  禁止使用 <script> 标签动态注入内容，禁止使用 innerHTML 或 getElementById
  等方式填充数据
- HTML 文件自身必须是完整且自包含的，不依赖任何外部脚本

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


SYSTEM_PROMPT = """你是一个专业的 AI 与机器人领域学术新闻编辑。
你的任务是根据搜索素材生成完整的 HTML 周报。

核心原则：
1. 只保留 7 天内的新闻（即使用户素材中混入了更早的，也要筛掉）
2. 优先学术论文和技术突破，产业政策为辅
3. 所有内容翻译为简体中文，保留原始 URL
4. 严格输出纯 HTML（从 <!DOCTYPE html> 开始），不要 markdown 代码块包裹
5. 绝对不要使用任何 JavaScript——所有内容都必须是静态 HTML 标签

HTML 要求：
- 内嵌 <style>，白底主题
- 每个板块一个 <div class="section">
- 每条新闻一个 <div class="card">，包含标题、日期、摘要、标签、来源链接
- 顶部统计栏：总条目数、板块数
- 如果某个板块确实没有本周相关新闻，标注"本周暂无相关重大进展"

CSS 变量参考（白底主题）：
--bg: #ffffff; --card-bg: #f8f9fb; --text: #1a1a2e; --text-secondary: #555;
--accent: #2563eb; --border: #e2e5ea; 标签用浅色底+对应色字"""


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def generate_html(results_text: str, dr: dict, deepseek: OpenAI) -> str:
    """调用 DeepSeek 生成完整 HTML。返回清洗后的纯 HTML 字符串。"""
    print("调用 DeepSeek API 生成周报 HTML ...", flush=True)

    resp = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(results_text, dr)},
        ],
        temperature=0.3,
        max_tokens=16384,
    )
    content = resp.choices[0].message.content or ""

    # 去掉 <!DOCTYPE 或 <html 之前的噪音
    doctype_pos = content.find("<!DOCTYPE")
    html_pos = content.find("<html")
    cut = -1
    if doctype_pos >= 0:
        cut = doctype_pos
    elif html_pos >= 0:
        cut = html_pos
    if cut > 0:
        content = content[cut:]

    # 去掉可能的 markdown 代码块包裹
    lines = content.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    content = "\n".join(lines)

    return content.strip()


# ---------------------------------------------------------------------------
# index.html 静态化
# ---------------------------------------------------------------------------

INDEX_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 与机器人周报</title>
<style>
  :root {{
    --bg: #ffffff; --card-bg: #f8f9fb; --text: #1a1a2e;
    --text-secondary: #555; --accent: #2563eb; --border: #e2e5ea;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
    max-width: 720px; margin: 0 auto; padding: 40px 24px;
  }}
  .header {{ text-align: center; padding: 32px 0; border-bottom: 2px solid var(--border); margin-bottom: 32px; }}
  .header h1 {{ font-size: 26px; font-weight: 800; }}
  .header h1 span {{ color: var(--accent); }}
  .header p {{ color: var(--text-secondary); margin-top: 6px; font-size: 14px; }}
  .report-list {{ list-style: none; }}
  .report-list li {{ margin-bottom: 10px; }}
  .report-list a {{
    display: block; padding: 14px 18px;
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 10px; text-decoration: none; color: var(--text);
    transition: box-shadow 0.15s;
  }}
  .report-list a:hover {{ box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
  .report-list .date {{ font-weight: 700; font-size: 16px; }}
  .report-list .label {{ font-size: 12px; color: var(--text-secondary); }}
  .latest-badge {{
    display: inline-block; background: var(--accent); color: #fff;
    font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 8px;
    vertical-align: middle;
  }}
  .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); font-size: 13px; color: var(--text-secondary); }}
  .footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
<div class="header">
  <h1>AI 与机器人<span>周报</span></h1>
  <p>每周一 9:00（北京时间）自动生成 · 学术论文与技术突破 · 中英双源</p>
</div>
<ul class="report-list">
{items}
</ul>
<div class="footer">
  <p>由 GitHub Actions 自动生成 · <a href="https://github.com/Practitioner108/ai-robotics-weekly">查看仓库</a></p>
  <p style="margin-top:4px;">最后更新: {updated_at}</p>
</div>
</body>
</html>"""


def update_index_html(dr: dict) -> None:
    """扫描所有已生成报告，重写纯静态 index.html。"""
    report_dir = OUTPUT_DIR
    html_files: list[str] = []

    if os.path.isdir(report_dir):
        for fname in os.listdir(report_dir):
            if fname.endswith(".html"):
                html_files.append(fname)

    # 按文件名（日期）降序排列
    html_files.sort(reverse=True)

    items_parts: list[str] = []
    for i, fname in enumerate(html_files):
        d = fname.replace(".html", "")
        badge = '<span class="latest-badge">最新</span>' if i == 0 else ""
        # 路径中用原始中文名，现代浏览器完全支持
        items_parts.append(
            f'    <li><a href="{OUTPUT_DIR}/{fname}">\n'
            f'      <span class="date">{d}</span>{badge}\n'
            f'      <span class="label"> · 点击查看</span>\n'
            f'    </a></li>'
        )

    if not items_parts:
        items_parts.append(
            '    <li style="color:var(--text-secondary);text-align:center;padding:20px;">'
            '暂无报告</li>'
        )

    updated_at = dr["today"].strftime("%Y-%m-%d %H:%M UTC")
    index_html = INDEX_HTML_TEMPLATE.format(
        items="\n".join(items_parts),
        updated_at=updated_at,
    )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

    print(f"index.html 已更新（{len(html_files)} 份报告）", flush=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    deepseek, tavily = _init_clients()
    dr = _compute_date_range()
    sections = _build_sections(dr)

    print("=" * 50)
    print("AI 与机器人周报生成器")
    print(f"日期范围  : {dr['range_cn']}")
    print(f"输出文件  : {OUTPUT_DIR}/{dr['iso_date']}.html")
    print("=" * 50)
    print()

    # ---- 第一步：搜索 ----
    print("--- 第一步：Tavily 搜索 ---")
    all_results = search_all(sections, tavily)
    total = sum(len(v) for v in all_results.values())
    print(f"\n  共获取 {total} 条搜索结果（去重后）\n")

    if total < MIN_SEARCH_RESULTS:
        print(
            f"错误: 搜索结果不足 ({total} < {MIN_SEARCH_RESULTS})，"
            f"可能 API 异常或网络故障，放弃生成。",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- 第二步：LLM 生成 ----
    results_text = format_results(all_results)

    print("--- 第二步：DeepSeek 生成 HTML ---")
    html = generate_html(results_text, dr, deepseek)
    print(f"  HTML 长度: {len(html)} 字符")

    if not html or len(html.encode("utf-8")) < MIN_HTML_BYTES:
        print(
            f"错误: 生成的 HTML 过短 ({len(html)} 字符 / "
            f"{len(html.encode('utf-8'))} 字节 < {MIN_HTML_BYTES})，"
            f"大概率是 API 返回异常，放弃写入。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 结构完整性校验（真正的门禁）
    if "<!DOCTYPE html>" not in html and "<!DOCTYPE" not in html:
        print("错误: 生成的 HTML 缺少 DOCTYPE 声明，内容可能不完整。", file=sys.stderr)
        sys.exit(1)

    if "</html>" not in html:
        print("错误: 生成的 HTML 缺少 </html> 闭合标签，可能被截断。", file=sys.stderr)
        sys.exit(1)

    # 软警告：内容密度偏低时不阻断，但在日志中提示
    section_count = html.count('class="section"') + html.count("class='section'")
    card_count = html.count('class="card"') + html.count("class='card'")
    if section_count < 3:
        print(f"⚠️  警告: 仅检测到 {section_count} 个板块，内容可能偏少（非致命）", flush=True)
    if card_count < 3:
        print(f"⚠️  警告: 仅检测到 {card_count} 条新闻卡片，本周可能是淡周（非致命）", flush=True)

    print()

    # ---- 第三步：保存报告 ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{dr['iso_date']}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"--- 周报已保存至 {output_path} ---")

    # ---- 第四步：更新 index.html ----
    update_index_html(dr)

    # ---- 第五步：回传日期给 GitHub Actions ----
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as f:
            f.write(f"report_date={dr['iso_date']}\n")


if __name__ == "__main__":
    main()
