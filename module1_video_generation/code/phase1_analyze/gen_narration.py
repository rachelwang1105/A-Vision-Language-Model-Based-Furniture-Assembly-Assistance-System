"""
gen_narration.py — 用 GPT-4o 分析說明書圖片，生成每步驟的 TTS 旁白

執行：python3 code/phase1_analyze/gen_narration.py

輸出：
  output/pnp/narration_step01.txt ... narration_step04.txt
"""

import base64
import json
import os
import platform
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
else:
    BASE_DIR = r"C:\Project"

IMG_DIR     = os.path.join(BASE_DIR, "image", "steps_crop", "vittsjo_2")
PNP_DIR     = os.path.join(BASE_DIR, "output", "pnp")
PARTS_JSON  = os.path.join(BASE_DIR, "dataset", "small_parts.json")
STEPS       = ["step01", "step02", "step03", "step04"]

client = OpenAI()

small_parts_data = json.load(open(PARTS_JSON, encoding="utf-8"))


def gen_narration(step: str) -> str:
    step_num = step.replace("step", "")
    img_path = os.path.join(IMG_DIR, f"{step}.png")
    with open(img_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()

    # 取得零件類型描述（不含料號）
    parts = small_parts_data.get(step, [])
    parts_desc = "、".join(
        f"{p['name'].split('（')[0]}×{p['count']}"  # 去掉括號內料號
        for p in parts
    ) if parts else "無小零件"

    prompt = f"""這是 IKEA VITTSJO_2 玻璃桌組裝說明書第 {step_num} 步的圖片。
本步驟用到的零件（類型）：{parts_desc}。

請根據圖片中的動作，寫一段 45～60 字的繁體中文語音旁白，說明這個步驟的組裝動作。

要求：
- 口語化、自然流暢，適合 TTS 朗讀，聽起來像真人講解
- 說明「做什麼動作」「放在哪裡」「確認什麼」，完整描述這步驟
- 不要講料號或數字編號（例如不要說 107616 之類的）
- 不要加標點以外的格式
- 只輸出旁白文字本身，不要解釋"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def main():
    for step in STEPS:
        img_path = os.path.join(IMG_DIR, f"{step}.png")
        if not os.path.exists(img_path):
            print(f"⚠️  找不到 {img_path}，跳過")
            continue

        print(f"[{step}] 生成旁白…")
        text = gen_narration(step)
        print(f"  → {text}（{len(text)} 字）")

        out = os.path.join(PNP_DIR, f"narration_{step}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  💾 {out}")


if __name__ == "__main__":
    main()
