"""
step_anim.py — 單一步驟場景，在 Blender Scripting 介面執行

step01：零件靜態放在目標位置
step02~04：新零件從 connection 推算的插入方向飛入
            base 零件（上一步已有）靜態放在組裝位置
step05：靜態，全部在原點

CLI（背景渲染）：
    blender --background --python step_anim.py -- step02
"""
import bpy
import math
import os
import sys
import json
import platform
import mathutils
from bpy_extras.object_utils import world_to_camera_view

# ==========================================
# 設定
# ==========================================
# CLI 優先：blender --background --python step_anim.py -- stepXX
_cli_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
STEP           = _cli_args[0] if _cli_args else "step01"   # ← GUI 時改這裡
START_DISTANCE = 1.5
CAMERA_PULL_BACK = 0.0

GLASS_PARTS = {"3"}   # Part ID 對應玻璃層板

CAMERA_POSE_FILES = {
    "step01": "camera_pose_step01.json",
    "step02": "camera_pose_step02.json",
    "step03": "camera_pose_step03.json",
    "step04": "camera_pose_step04.json",
    "step05": "camera_pose_step05_ex.json",
}
ROLL_180 = {
    "step01": False, "step02": False,
    "step03": False, "step04": False, "step05": False,
}
frames_total = 180

# ==========================================
# 路徑
# ==========================================
if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
else:
    BASE_DIR = r"C:\Project"

PARTS_DIR      = os.path.join(BASE_DIR, "dataset", "parts", "Misc", "vittsjo_2")
JSON_PATH      = os.path.join(BASE_DIR, "dataset", "main_data.json")
STEPS_DIR      = os.path.join(BASE_DIR, "code", "phase2_align", "step")
PNP_DIR        = os.path.join(BASE_DIR, "output", "pnp")
FURNITURE_NAME = "vittsjo_2"

# ==========================================
# 讀取步驟設定（判斷新零件 vs base 零件）
# ==========================================
all_step_names = sorted([f.replace(".json","") for f in os.listdir(STEPS_DIR) if f.endswith(".json")])
step_configs   = {n: json.load(open(os.path.join(STEPS_DIR, f"{n}.json"))) for n in all_step_names}

step_cfg      = step_configs[STEP]
show_parts    = step_cfg["show_parts"]
end_step_id   = step_cfg.get("dataset_step_id")
start_step_id = step_cfg.get("anim_start_step_id")
is_animated   = start_step_id is not None

cur_idx       = all_step_names.index(STEP)
prev_show     = step_configs[all_step_names[cur_idx - 1]]["show_parts"] if cur_idx > 0 else []
new_parts     = [p for p in show_parts if p not in prev_show]
base_parts    = [p for p in show_parts if p in prev_show]
print(f"{STEP}  new={new_parts}  base={base_parts}  animated={is_animated}")

# ==========================================
# 讀取連接關係
# ==========================================
all_data  = json.load(open(JSON_PATH))
furniture = next((x for x in all_data if x.get("name") == FURNITURE_NAME), None)

def get_ds_step(ds_id):
    if ds_id is None or not furniture:
        return None
    return next((s for s in furniture["steps"] if s["step_id"] == ds_id), None)

# 收集連接關係（start 優先，保持順序，讓細粒度的個別連接先被比對）
connections_raw = []
for ds_id in [start_step_id, end_step_id]:   # list 保持順序，不用 set
    s = get_ds_step(ds_id)
    if s:
        connections_raw.extend(s.get("connections", []))

# ==========================================
# 清空場景 & 建立相機
# ==========================================
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

cam_data = bpy.data.cameras.new("Camera")
cam_obj  = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.rotation_mode = 'QUATERNION'

pose_file = CAMERA_POSE_FILES.get(STEP)
if pose_file:
    pp = os.path.join(PNP_DIR, pose_file)
    if os.path.exists(pp):
        pose     = json.load(open(pp))
        R_mat    = mathutils.Matrix(pose["R_cam_to_world"]).to_4x4()
        R_mat.translation = mathutils.Vector(pose["camera_location"])
        if ROLL_180.get(STEP):
            R_mat = R_mat @ mathutils.Matrix.Rotation(math.pi, 4, 'Z')
        cam_obj.matrix_world = R_mat
        cam_obj.data.lens    = pose["intrinsics"]["focal_mm_blender"]
        if CAMERA_PULL_BACK != 0.0:
            back = cam_obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
            cam_obj.location += back * CAMERA_PULL_BACK
        bpy.context.scene.render.resolution_x = pose["intrinsics"]["width"]
        bpy.context.scene.render.resolution_y = pose["intrinsics"]["height"]
        print(f"相機：{pose_file}  pull_back={CAMERA_PULL_BACK}")
    else:
        print(f"⚠️  找不到 {pose_file}")

# ==========================================
# 匯入零件
# ==========================================
parts_dict = {}
for fn in sorted(os.listdir(PARTS_DIR)):
    if not fn.endswith(".obj"):
        continue
    pid = fn.replace(".obj", "").lstrip("0") or "0"
    bpy.ops.wm.obj_import(filepath=os.path.join(PARTS_DIR, fn))
    if not bpy.context.selected_objects:
        continue
    obj = bpy.context.selected_objects[0]
    obj.name = f"Part_{pid}"
    obj.rotation_mode = 'XYZ'
    if pid not in show_parts:
        obj.hide_viewport = True
        obj.hide_render   = True
    parts_dict[pid] = obj

# ==========================================
# 計算重心（模型空間，不套外參旋轉）
# ==========================================
def calc_centroid_model(part_ids):
    """模型空間重心（v.co，不套任何矩陣）
    所有零件 end 位置統一用原點（identity），方向也用模型空間，保持同一坐標系。"""
    total = mathutils.Vector((0, 0, 0))
    count = 0
    for pid in part_ids:
        if pid not in parts_dict:
            continue
        for v in parts_dict[pid].data.vertices:
            total += v.co
            count += 1
    return (total / count) if count else mathutils.Vector((0, 0, 0))

def parse_ids(s):
    return [p.strip() for p in str(s).split(",")]

# ==========================================
# 計算插入方向（從 connections，模型空間）
# ==========================================
def insertion_direction(part_id):
    """
    在 connections 裡找到此零件的連接對象，
    回傳方向向量：從連接對象重心 → 自己重心（歸一化，模型空間）
    零件沿這個方向的反向飛入，最後插入到位
    """
    my_c = calc_centroid_model([part_id])
    for conn in connections_raw:
        a_ids = parse_ids(conn[0])
        b_ids = parse_ids(conn[1])
        if part_id in a_ids:
            partner_ids = b_ids
        elif part_id in b_ids:
            partner_ids = a_ids
        else:
            continue
        # 只考慮 show_parts 裡的 partner（排除不在場景裡的零件）
        partner_ids = [p for p in partner_ids if p in show_parts]
        if not partner_ids:
            continue
        partner_c = calc_centroid_model(partner_ids)
        d = my_c - partner_c
        if d.length > 1e-4:
            print(f"    Part_{part_id} 插入方向：{tuple(round(x,3) for x in d.normalized())}  (連接到 {partner_ids})")
            return d.normalized()
    # fallback：從上方飛入
    print(f"    Part_{part_id} 無連接資料，預設從上方飛入")
    return mathutils.Vector((0, 0, 1))


# ==========================================
# 放置零件 / 建立動畫（全部使用原點，不套外參旋轉）
# ==========================================
for pid, obj in parts_dict.items():
    if pid not in show_parts:
        continue

    do_anim = is_animated and (pid in new_parts)

    if not do_anim:
        # 靜態放置（原點）
        obj.location = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)
    else:
        # 動畫：沿插入方向反向出發 → 飛入原點
        d = insertion_direction(pid)

        obj.location = mathutils.Vector((0, 0, 0)) + d * START_DISTANCE
        obj.rotation_euler = (0, 0, 0)
        obj.keyframe_insert(data_path="location",       frame=1)
        obj.keyframe_insert(data_path="rotation_euler", frame=1)

        obj.location = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)
        obj.keyframe_insert(data_path="location",       frame=frames_total)
        obj.keyframe_insert(data_path="rotation_euler", frame=frames_total)

# ==========================================
# 場景設定
# ==========================================
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = frames_total
bpy.context.scene.frame_set(frames_total)   # 移到結束幀讓 matrix_world 反映 end 位置
bpy.context.view_layer.update()

bpy.context.scene.frame_set(1)

# ==========================================
# 材質
# ==========================================
def _get_bsdf(mat):
    """依 type 找 Principled BSDF，找不到就新建一個。"""
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

def _set_input(bsdf, name, value):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = value

def make_metal_mat():
    mat = bpy.data.materials.new("Mat_Metal")
    mat.use_nodes = True
    bsdf = _get_bsdf(mat)
    _set_input(bsdf, "Base Color",  (0.08, 0.08, 0.08, 1.0))
    _set_input(bsdf, "Metallic",    0.9)
    _set_input(bsdf, "Roughness",   0.35)
    return mat

def make_glass_mat():
    mat = bpy.data.materials.new("Mat_Glass")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = True
    bsdf = _get_bsdf(mat)
    _set_input(bsdf, "Base Color",         (0.85, 0.92, 1.0, 1.0))
    _set_input(bsdf, "Transmission Weight", 1.0)   # Blender 4.x
    _set_input(bsdf, "Transmission",        1.0)   # Blender 3.x fallback
    _set_input(bsdf, "IOR",                1.52)
    _set_input(bsdf, "Roughness",          0.04)
    _set_input(bsdf, "Alpha",              0.15)
    return mat


def make_wood_mat():
    """深咖啡色木紋地板材質（程序紋理，不需圖片）"""
    mat = bpy.data.materials.new("Mat_Wood")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out     = nodes.new("ShaderNodeOutputMaterial")
    bsdf    = nodes.new("ShaderNodeBsdfPrincipled")
    ramp    = nodes.new("ShaderNodeValToRGB")
    wave    = nodes.new("ShaderNodeTexWave")
    noise   = nodes.new("ShaderNodeTexNoise")
    mix_rgb = nodes.new("ShaderNodeMixRGB")
    coord   = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")

    # 木紋年輪色帶：深棕 → 中棕
    ramp.color_ramp.elements[0].color    = (0.12, 0.06, 0.02, 1.0)
    ramp.color_ramp.elements[1].color    = (0.48, 0.24, 0.09, 1.0)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = 1.0

    # 波紋（木紋年輪）
    wave.wave_type            = 'RINGS'
    wave.inputs['Scale'].default_value       = 6.0
    wave.inputs['Distortion'].default_value  = 3.5
    wave.inputs['Detail'].default_value      = 8.0
    wave.inputs['Detail Scale'].default_value = 2.0
    wave.inputs['Detail Roughness'].default_value = 0.6

    # 細粒雜訊疊加
    noise.inputs['Scale'].default_value     = 30.0
    noise.inputs['Detail'].default_value    = 6.0
    noise.inputs['Roughness'].default_value = 0.6

    mix_rgb.blend_type = 'MULTIPLY'
    mix_rgb.inputs['Fac'].default_value = 0.25

    # 拉伸 mapping（讓木紋沿 X 方向延伸）
    mapping.inputs['Scale'].default_value = (4.0, 1.0, 1.0)

    links.new(coord.outputs['Object'],    mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'],  wave.inputs['Vector'])
    links.new(mapping.outputs['Vector'],  noise.inputs['Vector'])
    links.new(wave.outputs['Color'],      ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'],      mix_rgb.inputs['Color1'])
    links.new(noise.outputs['Color'],     mix_rgb.inputs['Color2'])
    links.new(mix_rgb.outputs['Color'],   bsdf.inputs['Base Color'])

    bsdf.inputs['Roughness'].default_value  = 0.55
    bsdf.inputs['Specular IOR Level'].default_value = 0.3 if 'Specular IOR Level' in bsdf.inputs else 0
    if 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.3

    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

metal_mat = make_metal_mat()
glass_mat = make_glass_mat()

for pid, obj in parts_dict.items():
    if pid not in show_parts:
        continue
    mat = glass_mat if pid in GLASS_PARTS else metal_mat
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# ==========================================
# 打光
# ==========================================
def add_area_light(name, energy, color, location, rotation_euler, size=1.0):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.color  = color
    light.data.size   = size
    light.rotation_euler = rotation_euler
    return light

# 世界背景白色
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value    = (0.62, 0.42, 0.22, 1.0)  # 咖啡木色
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55

# 主光（左前上方）
add_area_light("Light_Key",  800, (1.0, 0.97, 0.90),
               location=(1.5, -1.5, 2.5),
               rotation_euler=(math.radians(45), 0, math.radians(-45)),
               size=1.5)
# 補光（右側，柔和）
add_area_light("Light_Fill", 200, (0.85, 0.90, 1.0),
               location=(-1.8, -0.5, 1.0),
               rotation_euler=(math.radians(20), 0, math.radians(110)),
               size=2.0)
# 背光（後上，勾勒輪廓）
add_area_light("Light_Rim",  400, (1.0, 1.0, 1.0),
               location=(0.0, 2.0, 2.0),
               rotation_euler=(math.radians(-50), 0, math.radians(180)),
               size=1.0)

# ==========================================
# EEVEE 渲染設定 & 輸出
# ==========================================
scene = bpy.context.scene
# Blender 4.2+ 用 EEVEE_NEXT，舊版用 BLENDER_EEVEE
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
eevee = scene.eevee
for attr, val in [("use_bloom", True), ("use_ambient_occlusion", True),
                  ("use_ssr", True), ("ssr_quality", 0.5), ("taa_render_samples", 64)]:
    if hasattr(eevee, attr):
        setattr(eevee, attr, val)

scene.render.fps = 24
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode  = "RGBA"
scene.render.image_settings.compression = 15   # 0-100，低壓縮速度快

RENDERS_DIR = os.path.join(BASE_DIR, "output", "renders", STEP)
os.makedirs(RENDERS_DIR, exist_ok=True)
scene.render.filepath = os.path.join(RENDERS_DIR, "frame_")

# ==========================================
# 渲染動畫
# ==========================================
print(f"\n🎬 開始渲染 {STEP} → {scene.render.filepath}")
bpy.ops.render.render(animation=True)
print(f"✅ 渲染完成")

print(f"\n✅ {STEP}  {'動畫' if is_animated else '靜態'}  幀數 1–{frames_total}")
print(f"   零件放置：原點（模型空間，無外參旋轉）")
