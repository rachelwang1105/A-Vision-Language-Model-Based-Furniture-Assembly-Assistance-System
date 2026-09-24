"""
pick_2d_points.py  ─ 一般 Python 執行
在說明書圖上點選特徵點，儲存 2D 座標。

操作：
  左鍵    點選特徵點（標記編號）
  右鍵    刪除最後一個點
  Enter   儲存並結束
  ESC     放棄並結束

需要：pip install opencv-python
"""

import cv2
import json
import os
import platform

if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
else:
    BASE_DIR = r"C:\Project"

STEP        = "step04"   # ← 改這裡換步驟 (step01 / step02 / step03 / step04)
IMG_PATH    = os.path.join(BASE_DIR, "image", "steps_crop", "vittsjo_2", f"{STEP}.png")
OUT_PATH    = os.path.join(BASE_DIR, "output", "pnp", f"points_2d_{STEP}.json")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── 狀態 ──────────────────────────────────────────────────────────────────────
img_orig = cv2.imread(IMG_PATH)
if img_orig is None:
    raise FileNotFoundError(f"找不到圖片：{IMG_PATH}")

points = []   # [(x, y), ...]

def redraw():
    canvas = img_orig.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(canvas, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    h, w = canvas.shape[:2]
    hint = f"Points: {len(points)}  |  L-click=add  R-click=undo  Enter=save  ESC=quit"
    cv2.putText(canvas, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 200), 1)
    cv2.imshow("Pick 2D Points", canvas)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw()

cv2.namedWindow("Pick 2D Points")
cv2.setMouseCallback("Pick 2D Points", on_mouse)
redraw()

while True:
    key = cv2.waitKey(0)
    if key == 13:   # Enter
        with open(OUT_PATH, "w") as f:
            json.dump({"step": STEP, "image": IMG_PATH, "points_2d": points}, f, indent=2)
        print(f"儲存 {len(points)} 個點 → {OUT_PATH}")
        break
    elif key == 27:  # ESC
        print("放棄，未儲存。")
        break

cv2.destroyAllWindows()
