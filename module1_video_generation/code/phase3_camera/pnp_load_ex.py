"""
在 Blender 執行：用外參把零件擺到正確位置，再用 pick_3d_points.py 選點

步驟：
  1. 執行本腳本（MODE="load"）→ 零件依外參到位
  2. 切換到 pick_3d_points.py（STEP 同步改好），MODE="record" 逐點選
  3. MODE="save" 儲存

STEP 對應 dataset_step_id（從 step/*.json 讀取）：
  step01 → dataset_step_id 0
  step02 → dataset_step_id 0
  step03 → dataset_step_id 1
  step04 → dataset_step_id 2
"""
import bpy, json, os, platform, mathutils

BASE_DIR   = os.path.expanduser("~/Documents/Project/project") if platform.system()=="Darwin" else r"C:\Project"
STEP       = "step02"   # ← 改這裡換步驟 (step01 / step02 / step03 / step04)
PARTS_DIR  = os.path.join(BASE_DIR, "dataset", "parts", "Misc", "vittsjo_2")
JSON_PATH  = os.path.join(BASE_DIR, "dataset", "main_data.json")
STEPS_DIR  = os.path.join(BASE_DIR, "code", "phase2_align", "step")

# ── 讀取該說明書步驟的 dataset_step_id ──────────────────────────────
step_json_path = os.path.join(STEPS_DIR, f"{STEP}.json")
if not os.path.exists(step_json_path):
    raise FileNotFoundError(f"找不到 {step_json_path}，請確認 step/ 資料夾有此檔案")
step_cfg = json.load(open(step_json_path))
ds_id    = step_cfg.get("dataset_step_id")
print(f"說明書步驟 {STEP} → dataset_step_id = {ds_id}")

# ── 讀取外參 ─────────────────────────────────────────────────────────
all_data       = json.load(open(JSON_PATH))
furniture_data = next((x for x in all_data if x.get("name")=="vittsjo_2"), None)
if not furniture_data:
    raise Exception("找不到 vittsjo_2")

pose_map = {}
if ds_id is not None:
    for step in furniture_data["steps"]:
        if step["step_id"] == ds_id:
            for i, part_group in enumerate(step["parts"]):
                matrix = step["extrinsics"][i]
                for pid in str(part_group).split(","):
                    pose_map[pid.strip()] = matrix
            break
    print(f"外參零件：{list(pose_map.keys())}")
else:
    print("dataset_step_id=null，零件將放在原點")

# ── 清空場景並載入零件 ───────────────────────────────────────────────
if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

count = 0
show_parts = step_cfg.get("show_parts", [])
for fn in sorted(os.listdir(PARTS_DIR)):
    if not fn.endswith(".obj"):
        continue
    part_id = fn.replace(".obj", "").lstrip("0") or "0"
    bpy.ops.wm.obj_import(filepath=os.path.join(PARTS_DIR, fn))
    if not bpy.context.selected_objects:
        continue
    obj = bpy.context.selected_objects[0]
    obj.name = f"Part_{part_id}"

    if part_id in pose_map:
        obj.matrix_world = mathutils.Matrix(pose_map[part_id])
    else:
        obj.location       = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)

    # 不在這步驟顯示的零件先藏起來（方便選點時畫面乾淨）
    obj.hide_viewport = part_id not in show_parts
    count += 1

print(f"✅ 載入 {count} 個零件（顯示：{show_parts}）")
print(f"   接著執行 pick_3d_points.py（STEP='{STEP}', MODE='record'）開始選點")
