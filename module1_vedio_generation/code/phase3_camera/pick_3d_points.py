"""
pick_3d_points.py  ─ 在 Blender Scripting 介面執行
每次執行記錄一個頂點，重複執行直到收集完所有點。

流程：
  1. Edit Mode 選取對應說明書「點 0」的頂點
  2. 執行本腳本 → 記錄點 0
  3. Edit Mode 選取「點 1」的頂點
  4. 執行本腳本 → 記錄點 1
  5. 重複直到所有點都記錄完
  6. 改 MODE = "save" 再執行一次 → 儲存 JSON

修改頂部的 MODE 控制行為：
  "record" → 記錄目前選取的頂點
  "undo"   → 刪除最後一個點
  "status" → 顯示目前所有點
  "save"   → 儲存並結束
  "reset"  → 清空重來
  "load"   → 載入 vittsjo_2 零件到場景
"""

import bpy
import json
import mathutils
import os
import platform

# ── 設定 ──────────────────────────────────────────────────────────────────────
MODE = "record"   # 改這裡：record / undo / status / save / reset / load

if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
else:
    BASE_DIR = r"C:\Project"

STEP      = "step02"   # ← 改這裡換步驟 (step01 / step02 / step03 / step04)
TAG       = "_ex"      # ← 檔名後綴："" = 原始標點，"_ex" = 外參標點
TMP_PATH  = os.path.join(BASE_DIR, "output", "pnp", f"_tmp_3d_{STEP}{TAG}.json")
OUT_PATH  = os.path.join(BASE_DIR, "output", "pnp", f"points_3d_{STEP}{TAG}.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# 從暫存檔讀取目前累積的點
if os.path.exists(TMP_PATH):
    with open(TMP_PATH) as f:
        points_3d = json.load(f)
else:
    points_3d = []

# ── 執行 ──────────────────────────────────────────────────────────────────────
if MODE == "record":
    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        print("❌ 請先選取一個 Mesh 物件")
    else:
        bpy.ops.object.mode_set(mode='OBJECT')
        selected = [v for v in obj.data.vertices if v.select]
        bpy.ops.object.mode_set(mode='EDIT')

        if len(selected) == 0:
            print("❌ 沒有選取頂點，請在 Edit Mode 選取頂點")
        else:
            import mathutils
            avg = mathutils.Vector((0, 0, 0))
            for v in selected:
                avg += obj.matrix_world @ v.co
            avg /= len(selected)
            pt = [round(avg.x, 6), round(avg.y, 6), round(avg.z, 6)]
            points_3d.append(pt)
            with open(TMP_PATH, "w") as f:
                json.dump(points_3d, f)
            print(f"✅ 點 {len(points_3d)-1} 記錄：{pt}  （選了 {len(selected)} 個頂點取平均，共 {len(points_3d)} 個）")

elif MODE == "undo":
    if points_3d:
        removed = points_3d.pop()
        with open(TMP_PATH, "w") as f:
            json.dump(points_3d, f)
        print(f"↩️  刪除點 {len(points_3d)}：{removed}")
    else:
        print("沒有點可以刪除")

elif MODE == "status":
    print(f"目前共 {len(points_3d)} 個點：")
    for i, pt in enumerate(points_3d):
        print(f"  {i}: {pt}")

elif MODE == "save":
    with open(OUT_PATH, "w") as f:
        json.dump({"step": STEP, "points_3d": points_3d}, f, indent=2)
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)
    print(f"💾 儲存 {len(points_3d)} 個點 → {OUT_PATH}")

elif MODE == "reset":
    points_3d = []
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)
    print("🗑️  已清空，重新開始")

elif MODE == "load":
    PARTS_DIR  = os.path.join(BASE_DIR, "dataset", "parts", "Misc", "vittsjo_2")
    STEPS_DIR  = os.path.join(BASE_DIR, "code", "phase2_align", "step")

    step_cfg   = json.load(open(os.path.join(STEPS_DIR, f"{STEP}.json")))
    show_parts = step_cfg.get("show_parts", [])

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    count = 0
    for fn in sorted(os.listdir(PARTS_DIR)):
        if not fn.endswith(".obj"):
            continue
        part_id = fn.replace(".obj", "").lstrip("0") or "0"
        bpy.ops.wm.obj_import(filepath=os.path.join(PARTS_DIR, fn))
        if not bpy.context.selected_objects:
            continue
        obj = bpy.context.selected_objects[0]
        obj.name          = f"Part_{part_id}"
        obj.location      = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)
        obj.hide_viewport = part_id not in show_parts
        count += 1

    print(f"✅ 載入 {count} 個零件（{STEP}），顯示：{show_parts}")
    print(f"   零件擺放：原點（模型空間，與 step_anim.py 一致）")
    print(f"   改 MODE = 'record' 開始選點")
