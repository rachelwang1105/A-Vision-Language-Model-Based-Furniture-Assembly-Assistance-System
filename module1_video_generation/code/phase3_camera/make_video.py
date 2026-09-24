"""
make_video.py — 後製處理腳本（普通 Python 執行）

輸入：
  output/renders/stepXX.mp4        Blender 渲染影片
  output/pnp/install_points_stepXX.json
  dataset/small_parts.json
  icons/                           零件圖示（PNG，建議帶透明通道）

輸出：
  output/final/stepXX_final.mp4

需要：pip install opencv-python edge-tts imageio-ffmpeg
"""

import base64
import cv2
import json
import os
import platform
import re
import subprocess
import asyncio
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── 路徑 ────────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
else:
    BASE_DIR = r"C:\Project"

RENDER_DIR   = os.path.join(BASE_DIR, "output", "renders")
PNP_DIR      = os.path.join(BASE_DIR, "output", "pnp")
ICONS_DIR    = os.path.join(BASE_DIR, "icons")
FINAL_DIR    = os.path.join(BASE_DIR, "output", "final")
PARTS_JSON   = os.path.join(BASE_DIR, "dataset", "small_parts.json")
FFMPEG       = imageio_ffmpeg.get_ffmpeg_exe()

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(os.path.join(FINAL_DIR, "audio"), exist_ok=True)

# ─── 設定 ─────────────────────────────────────────────────────────────
STEPS = ["step01", "step02", "step03", "step04"]

VOICE = "zh-TW-HsiaoChenNeural"
_openai_client = None

def _get_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def gen_narration(step: str, img_path: str, small_parts: list) -> str:
    """用 GPT-4o 分析說明書圖片，生成該步驟的繁體中文語音旁白。"""
    step_num = re.sub(r"\D", "", step)  # "step02" → "2"
    parts_desc = "、".join(
        f"{p['name']}×{p['count']}" for p in small_parts
    ) if small_parts else "無零件"

    with open(img_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()

    prompt = f"""這是 IKEA VITTSJO_2 玻璃桌組裝說明書第 {step_num} 步的圖片。
本步驟使用的零件：{parts_desc}。

請根據圖片內容，用繁體中文寫一段 15~30 字的語音旁白，說明這個步驟的動作。
要求：
- 口語化、自然流暢，適合 TTS 朗讀
- 不要加標點符號以外的任何說明
- 直接輸出旁白文字，不要多餘格式"""

    resp = _get_client().chat.completions.create(
        model="gpt-4o",
        max_tokens=100,
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

# 中文字體
_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "C:/Windows/Fonts/msjh.ttc",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
FONT_NAME  = ImageFont.truetype(_FONT_PATH, 20) if _FONT_PATH else ImageFont.load_default()
FONT_COUNT = ImageFont.truetype(_FONT_PATH, 28) if _FONT_PATH else ImageFont.load_default()
ANIM_FRAMES    = 180
INSTR_W      = 240
INSTR_MARGIN = 16
INSTR_ALPHA  = 0.88
INSTR_PAD    = 20    # 圓圈邊框外留白
IMG_DIR      = os.path.join(BASE_DIR, "image", "steps_crop", "vittsjo_2")

def detect_detail_crops(img_path):
    """偵測 IKEA 說明書右側所有細節圓＋矩形框，回傳 [(x,y,w,h),...] 依 y 排序。"""
    img = cv2.imread(img_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crops = []

    # ── 圓形偵測 ──────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
        minDist=int(min(h, w) * 0.2),
        param1=200, param2=0.75,
        minRadius=int(min(h, w) * 0.10),
        maxRadius=int(min(h, w) * 0.45),
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            if cx - r < w * 0.3:
                continue
            x1 = max(0, cx - r - INSTR_PAD)
            y1 = max(0, cy - r - INSTR_PAD)
            x2 = min(w, cx + r + INSTR_PAD)
            y2 = min(h, cy + r + INSTR_PAD)
            crops.append((x1, y1, x2 - x1, y2 - y1, cy))  # cy 用於排序

    # ── 矩形框偵測 ────────────────────────────────────
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < w * h * 0.015:
            continue
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        # 輪廓面積佔外接矩形面積的比例（矩形 ≈ 高填充率）
        fill = area / (rw * rh) if rw * rh else 0
        if fill < 0.55:            # 太不像矩形（圓形 fill ≈ π/4 ≈ 0.79，也可通過，但有圓形偵測負責）
            continue
        if rx + rw / 2 < w * 0.4: # 右側才要
            continue
        aspect = rw / rh if rh else 0
        if not (0.6 < aspect < 4.0):
            continue
        x1 = max(0, rx - INSTR_PAD)
        y1 = max(0, ry - INSTR_PAD)
        x2 = min(w, rx + rw + INSTR_PAD)
        y2 = min(h, ry + rh + INSTR_PAD)
        crops.append((x1, y1, x2 - x1, y2 - y1, ry + rh / 2))

    # 依 y 排序，去掉排序鍵
    crops.sort(key=lambda c: c[4])
    return [(x, y, cw, ch) for x, y, cw, ch, _ in crops]
PANEL_W        = 220
PANEL_ICON_SZ  = 52
PANEL_ALPHA    = 0.65


# ─── 工具函式 ─────────────────────────────────────────────────────────

def paste_icon(frame, icon, x, y):
    """貼上圖示（支援 RGBA 透明通道）"""
    h, w = icon.shape[:2]
    # 邊界保護
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    ix1, iy1 = x1 - x, y1 - y
    ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)
    roi  = frame[y1:y2, x1:x2].astype(np.float32)
    crop = icon[iy1:iy2, ix1:ix2]
    if crop.shape[2] == 4:
        a = crop[:, :, 3:4].astype(np.float32) / 255.0
        b = crop[:, :, :3].astype(np.float32)
        frame[y1:y2, x1:x2] = np.clip(b * a + roi * (1 - a), 0, 255).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = crop


def draw_parts_panel(frame, small_parts):
    """右下角零件圖示面板"""
    if not small_parts:
        return frame
    FH, FW = frame.shape[:2]
    rows      = len(small_parts)
    row_h     = PANEL_ICON_SZ + 16
    panel_h   = rows * row_h + 16
    x0        = FW - PANEL_W - 20
    y0        = FH - panel_h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + PANEL_W + 4, y0 + panel_h + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1.0 - PANEL_ALPHA, 0, frame)

    for i, part in enumerate(small_parts):
        ry = y0 + 8 + i * row_h

        # 圖示
        icon_path = os.path.join(ICONS_DIR, part.get("icon", ""))
        if os.path.exists(icon_path):
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                icon = cv2.resize(icon, (PANEL_ICON_SZ, PANEL_ICON_SZ))
                paste_icon(frame, icon, x0 + 8, ry)

        # 名稱 + 數量（Pillow 繪製中文）
        name  = part.get("name", "")
        count = f"×{part.get('count', 1)}"
        pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw  = ImageDraw.Draw(pil)
        tx = x0 + PANEL_ICON_SZ + 16
        draw.text((tx, ry + 4),  name,  font=FONT_NAME,  fill=(230, 230, 230))
        draw.text((tx, ry + 28), count, font=FONT_COUNT, fill=(0, 220, 255))
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return frame


async def _gen_audio(text, voice, path):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(path)


def generate_audio(step, img_path, small_parts):
    # 優先用 gen_install_points.py 生成的快取描述
    cache_path = os.path.join(PNP_DIR, f"narration_{step}.txt")
    if os.path.exists(cache_path):
        text = open(cache_path, encoding="utf-8").read().strip()
        print(f"  📝 旁白（快取）：{text}")
    else:
        text = gen_narration(step, img_path, small_parts)
        print(f"  📝 旁白：{text}")
    path = os.path.join(FINAL_DIR, "audio", f"{step}.mp3")
    asyncio.run(_gen_audio(text, VOICE, path))
    print(f"  🔊 語音：{path}")
    return path


# ─── 主流程 ───────────────────────────────────────────────────────────

def process_step(step):
    # 支援 PNG 序列（output/renders/stepXX/）或單一 MP4
    png_dir     = os.path.join(RENDER_DIR, step)
    mp4_path    = os.path.join(RENDER_DIR, f"{step}.mp4")
    use_png_seq = os.path.isdir(png_dir) and any(f.endswith(".png") for f in os.listdir(png_dir))

    if use_png_seq:
        frames_paths = sorted(
            os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")
        )
        if not frames_paths:
            print(f"⚠️  {png_dir} 裡沒有 PNG，跳過"); return
        first = cv2.imread(frames_paths[0])
        H, W  = first.shape[:2]
        total_f = len(frames_paths)
        fps     = 24.0
    elif os.path.exists(mp4_path):
        frames_paths = None
        cap     = cv2.VideoCapture(mp4_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 24.0
        W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        print(f"⚠️  找不到渲染來源：{png_dir} 或 {mp4_path}，跳過"); return

    # 載入零件資訊
    all_parts = json.load(open(PARTS_JSON)) if os.path.exists(PARTS_JSON) else {}
    small_parts = all_parts.get(step, [])

    print(f"\n[{step}]  {W}×{H}  {total_f}幀  {fps:.1f}fps  {'PNG序列' if use_png_seq else 'MP4'}")

    # 自動偵測說明書所有細節圓，裁切後依序堆疊在右上角
    instr_panels = []
    instr_path = os.path.join(IMG_DIR, f"{step}.png")
    if os.path.exists(instr_path):
        raw_full = cv2.imread(instr_path)
        if raw_full is not None:
            for x, y, cw, ch in detect_detail_crops(instr_path):
                patch    = raw_full[y:y+ch, x:x+cw]
                scaled_h = int(INSTR_W * ch / cw)
                instr_panels.append(cv2.resize(patch, (INSTR_W, scaled_h)))

    # 載入安裝點（screen_x/y 已是正確座標）
    ip_path = os.path.join(PNP_DIR, f"install_points_{step}.json")
    install_pts = []
    if os.path.exists(ip_path):
        install_pts = [p for p in json.load(open(ip_path)) if p.get("in_frame")]
    MARKER_START = max(0, total_f - 60)   # 最後 60 幀才顯示圓圈

    tmp_path = os.path.join(FINAL_DIR, f"_tmp_{step}.mp4")
    out      = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    for fi in range(total_f):
        if use_png_seq:
            frame = cv2.imread(frames_paths[fi])
            if frame is None:
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # 右上角細節示意圖（多張依序堆疊）
        cursor_y = INSTR_MARGIN
        for panel in instr_panels:
            ih, iw = panel.shape[:2]
            x1 = W - iw - INSTR_MARGIN
            y1 = cursor_y
            x2, y2 = x1 + iw, y1 + ih
            if x1 >= 0 and y2 <= H:
                roi = frame[y1:y2, x1:x2].astype(np.float32)
                frame[y1:y2, x1:x2] = np.clip(
                    panel.astype(np.float32) * INSTR_ALPHA + roi * (1 - INSTR_ALPHA), 0, 255
                ).astype(np.uint8)
            cursor_y = y2 + 8   # 8px 間隔

        # 零件面板
        frame = draw_parts_panel(frame, small_parts)

        # 黃色安裝點圓圈（最後 60 幀淡入，永遠在最前景）
        if install_pts and fi >= MARKER_START:
            alpha = min(1.0, (fi - MARKER_START) / 30.0)
            overlay = frame.copy()
            for pt in install_pts:
                cx, cy = int(pt["screen_x"]), int(pt["screen_y"])
                if 0 <= cx < W and 0 <= cy < H:
                    cv2.circle(overlay, (cx, cy), 10, (0, 255, 255), -1)  # 黃色實心點
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        out.write(frame)

    if not use_png_seq:
        cap.release()
    out.release()

    # 生成語音（LLM 根據說明書圖片分析內容）
    audio_path = generate_audio(step, instr_path, small_parts)

    # 合成影片 + 音訊：補 30 秒凍結幀，-shortest 讓語音決定結束點
    final_path = os.path.join(FINAL_DIR, f"{step}_final.mp4")
    subprocess.run([
        FFMPEG, "-y",
        "-i", tmp_path,
        "-i", audio_path,
        "-filter_complex",
        "[0:v]tpad=stop_mode=clone:stop_duration=30[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        final_path,
    ], check=True, capture_output=True)

    os.remove(tmp_path)
    print(f"  ✅ {final_path}")


if __name__ == "__main__":
    for step in STEPS:
        process_step(step)
    print("\n🎬 全部完成")
