"""
render_all.py — 自動渲染所有步驟

執行：python3 code/phase3_camera/render_all.py

每個步驟依序呼叫 Blender 背景模式執行 step_anim.py，
輸出到 output/renders/stepXX.mp4
"""

import subprocess
import sys
import os
import platform
import time

# ─── 路徑 ────────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
    BLENDER  = "/Applications/Blender.app/Contents/MacOS/Blender"
else:
    BASE_DIR = r"C:\Project"
    BLENDER  = r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"

SCRIPT = os.path.join(BASE_DIR, "code", "phase2_align", "step_anim.py")
STEPS  = ["step01", "step02", "step03", "step04"]

# ─── 執行 ─────────────────────────────────────────────────────────────
if not os.path.exists(BLENDER):
    print(f"❌ 找不到 Blender：{BLENDER}")
    sys.exit(1)

total_start = time.time()

for step in STEPS:
    print(f"\n{'='*50}")
    print(f"  渲染 {step}")
    print(f"{'='*50}")
    t0 = time.time()

    result = subprocess.run(
        [BLENDER, "--background", "--python", SCRIPT, "--", step],
        text=True,
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ {step} 完成（{elapsed:.0f}s）")
    else:
        print(f"  ❌ {step} 失敗（return code {result.returncode}）")

print(f"\n🎬 全部完成（總計 {time.time() - total_start:.0f}s）")
