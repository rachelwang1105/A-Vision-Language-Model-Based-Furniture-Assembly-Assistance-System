"""
gen_small_parts.py — 呼叫 Ollama vision 分析說明書圖片，輸出 small_parts.json

執行：python3 code/phase1_analyze/gen_small_parts.py

輸出：
  dataset/small_parts.json        make_video.py 使用的零件資訊
  icons/<name>.png                自動生成的簡易圖示
"""

import base64
import json
import os
import platform
import re
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── 路徑 ────────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
else:
    BASE_DIR = r"C:\Project"

IMG_DIR   = os.path.join(BASE_DIR, "image", "steps_crop", "vittsjo_2")
ICONS_DIR = os.path.join(BASE_DIR, "icons")
OUT_PATH  = os.path.join(BASE_DIR, "dataset", "small_parts.json")
os.makedirs(ICONS_DIR, exist_ok=True)

STEPS = ["step01", "step02", "step03", "step04"]

client = OpenAI()   # 讀取環境變數 OPENAI_API_KEY

# ─── LLM 分析 ────────────────────────────────────────────────────────

def analyze_small_parts(step: str) -> list:
    img_path = os.path.join(IMG_DIR, f"{step}.png")
    with open(img_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()

    prompt = """這是 IKEA 家具說明書的一個步驟圖片。

請先用文字說明你看到的內容，再輸出 JSON。

【第一步：觀察 Nx 重複次數】
- 圖中有「Nx」標示嗎（例如 4x）？記下這個數字 N。

【第二步：分析放大細節圖裡「單次動作」的用量】
- 圓圈放大圖中顯示的是「一個連接點」的安裝示範。
- ⚠️ 關鍵規則：IKEA 常用螺絲（如 107616）＋螺帽（如 107615）是「一組緊固件」，
  放大圖從兩端各畫一次（一端螺絲頭、另一端螺帽），看起來像「兩顆螺絲」，
  但實際上是「同一顆螺絲穿過孔洞，螺帽從另一端鎖上」= 只算 1 顆螺絲 + 1 個螺帽。
  判斷方式：若「料號不同但同時出現在同一個安裝示意」→ 每種料號只算 1 個。
- 操作示意圖（螺絲起子轉動、旋轉箭頭）是說明如何操作，不是額外零件。

【第三步：計算總數量】
總數量 = N × 每次動作的數量
例：4x 動作、每次用 1 顆螺絲（107616）+ 1 個螺帽（107615）→ 螺絲 4 顆、螺帽 4 個

【第四步：輸出 JSON】
排除大型結構件（框架、板材、立柱、玻璃桌板）。
螺絲和螺帽分開列。料號放括號內。
type：screw（螺絲）/ nut（螺帽/蝶形螺帽）/ foot（調整腳）/ pad / dowel / clip / other

```json
[
  {
    "name": "中文名稱（料號）",
    "icon": "英文小寫底線.png",
    "count": 整數,
    "type": "類型"
  }
]
```"""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }]
    )

    text = response.choices[0].message.content.strip()
    print(f"  [GPT-4o 推理]\n{text}\n")   # 印出推理過程方便除錯
    # 優先抓 ```json 區塊，fallback 抓第一個 [...]
    m = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
    if not m:
        m = re.search(r'(\[.*?\])', text, re.DOTALL)
    if not m:
        return []
    return json.loads(m.group(1))


# ─── 圖示生成 ────────────────────────────────────────────────────────

ICON_SZ = 64   # 像素

def make_icon(part_type: str, icon_name: str):
    """依零件類型生成簡易圖示 PNG（白底透明）"""
    img = np.zeros((ICON_SZ, ICON_SZ, 4), dtype=np.uint8)
    cx, cy, r = ICON_SZ // 2, ICON_SZ // 2, ICON_SZ // 2 - 6
    c = (255, 255, 255, 255)   # 白色

    if part_type == "screw":
        # 圓頭 + 柱身 + 十字槽
        cv2.circle(img, (cx, cy - r // 2), r // 2, c, -1)
        cv2.rectangle(img, (cx - 4, cy - r // 2), (cx + 4, cy + r), c, -1)
        cv2.line(img, (cx - 10, cy - r // 2), (cx + 10, cy - r // 2), (50, 50, 50, 255), 2)
        cv2.line(img, (cx, cy - r), (cx, cy - r // 4), (50, 50, 50, 255), 2)
    elif part_type == "nut":
        # 六角螺帽
        pts = []
        for i in range(6):
            angle = np.radians(60 * i - 30)
            pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
        cv2.fillPoly(img, [np.array(pts)], c)
        cv2.circle(img, (cx, cy), r // 3, (0, 0, 0, 255), -1)
    elif part_type == "foot":
        # 調整腳：大圓底 + 小圓柱
        cv2.circle(img, (cx, cy + r // 2), r // 2, c, -1)
        cv2.rectangle(img, (cx - 5, cy - r // 2), (cx + 5, cy + r // 2), c, -1)
        cv2.circle(img, (cx, cy - r // 2), r // 4, c, -1)
    elif part_type == "pad":
        # 橡皮墊：扁圓形
        cv2.ellipse(img, (cx, cy), (r, r // 3), 0, 0, 360, c, -1)
        cv2.ellipse(img, (cx, cy - r // 4), (r - 6, r // 5), 0, 0, 360, c, -1)
    elif part_type == "dowel":
        # 木榫：長方形 + 半圓頂部
        cv2.rectangle(img, (cx - 8, cy - r // 2), (cx + 8, cy + r), c, -1)
        cv2.ellipse(img, (cx, cy - r // 2), (8, 5), 0, 180, 360, c, -1)
    elif part_type == "clip":
        # C 形夾
        cv2.ellipse(img, (cx, cy), (r, r), 0, 30, 330, c, 6)
    elif part_type == "washer":
        # 環形墊片
        cv2.circle(img, (cx, cy), r, c, -1)
        cv2.circle(img, (cx, cy), r // 2, (0, 0, 0, 0), -1)
    else:
        # 預設：菱形
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]])
        cv2.fillPoly(img, [pts], c)

    path = os.path.join(ICONS_DIR, icon_name)
    cv2.imwrite(path, img)


# ─── 主流程 ──────────────────────────────────────────────────────────

def main():
    result = {}
    for step in STEPS:
        img_path = os.path.join(IMG_DIR, f"{step}.png")
        if not os.path.exists(img_path):
            print(f"⚠️  找不到 {img_path}，跳過")
            result[step] = []
            continue

        print(f"\n[{step}] 分析中…")
        try:
            parts = analyze_small_parts(step)
        except Exception as e:
            print(f"  ❌ LLM 錯誤：{e}")
            result[step] = []
            continue

        for p in parts:
            icon_name = p.get("icon", f"{p['name']}.png")
            p["icon"] = icon_name
            make_icon(p.get("type", "other"), icon_name)
            print(f"  ✔ {p['name']}  ×{p['count']}  → icons/{icon_name}")

        result[step] = parts

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {OUT_PATH}")


if __name__ == "__main__":
    main()
