"""
gen_contact_points_blender.py — BVH 接觸點偵測 (Blender Python)

將零件套用外參矩陣擺到組裝完成位置，用 BVHTree 找相鄰零件接觸面重心，
投影到對應步驟的渲染畫面座標，並在場景中放置 Empty 標記。

執行：
  /Applications/Blender.app/Contents/MacOS/Blender \
      --background --python code/phase1_analyze/gen_contact_points_blender.py
輸出：
  output/pnp/install_points_{step}.json
  contact_points.blend（含 Empty 標記）
"""

import bpy
import bmesh
import json
import os
import platform

import mathutils
from mathutils.bvhtree import BVHTree
import numpy as np

# ─── 路徑 ────────────────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/project")
else:
    BASE_DIR = r"C:\Project"

PARTS_DIR      = os.path.join(BASE_DIR, "dataset", "parts", "Misc", "vittsjo_2")
STEPS_DIR      = os.path.join(BASE_DIR, "code", "phase2_align", "step")
PNP_DIR        = os.path.join(BASE_DIR, "output", "pnp")
JSON_PATH      = os.path.join(BASE_DIR, "dataset", "main_data.json")
SMALL_PARTS    = os.path.join(BASE_DIR, "dataset", "small_parts.json")
DEBUG_DIR      = os.path.join(BASE_DIR, "output", "debug")
RENDER_DIR     = os.path.join(BASE_DIR, "output", "renders")

STEPS       = ["step01", "step02", "step03", "step04"]
# OBJ 已是組裝後相對位置（與相機標定座標系一致），不需套外參
# 單位：OBJ-imported Blender world units（≈ 0.05 = 5cm）
MAX_CONTACT = 0.05
RATIO       = 2.0
CLUSTER_GAP = 0.08
DEDUP_R     = 0.06

os.makedirs(PNP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ─── 讀取資料集 ───────────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    all_data = json.load(f)
furniture = next(x for x in all_data if x["name"] == "vittsjo_2")
connection_relation = furniture["connection_relation"]   # [[a,b], ...]

with open(SMALL_PARTS) as f:
    small_parts_data = json.load(f)   # {stepXX: [{type, count, ...}, ...]}

# ─── 步驟設定 ─────────────────────────────────────────────────────────────────
all_step_names = sorted(
    f.replace(".json", "") for f in os.listdir(STEPS_DIR) if f.endswith(".json")
)
cfgs = {
    n: json.load(open(os.path.join(STEPS_DIR, f"{n}.json")))
    for n in all_step_names
}

# ─── 清場 & 匯入零件 ─────────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

parts_objs: dict = {}
for fname in sorted(os.listdir(PARTS_DIR)):
    if not fname.endswith(".obj"):
        continue
    pid = fname.replace(".obj", "").lstrip("0") or "0"
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.wm.obj_import(filepath=os.path.join(PARTS_DIR, fname))
    obj = bpy.context.selected_objects[0]
    obj.name = f"Part_{pid}"
    # OBJ 已在組裝後座標系，保留 identity 位置（與相機標定座標系一致）
    parts_objs[pid] = obj

print(f"載入零件：{sorted(parts_objs.keys())}")

# ─── InstallPoints Collection ─────────────────────────────────────────────────
COLL_NAME = "InstallPoints"
ip_coll = bpy.data.collections.new(COLL_NAME)
bpy.context.scene.collection.children.link(ip_coll)


# ─── 工具函式 ─────────────────────────────────────────────────────────────────

def make_bvh(obj) -> BVHTree:
    """建立已套用 matrix_world 的 BVH 樹（以 BMesh 面構建）。"""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.normal_update()
    tree = BVHTree.FromBMesh(bm, epsilon=0.0)
    bm.free()
    return tree


def world_verts(obj) -> np.ndarray:
    """回傳世界座標頂點，shape=(N,3) float32。"""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    v = np.array([vert.co[:] for vert in bm.verts], dtype=np.float32)
    bm.free()
    return v


def contact_centroids_bvh(obj_new, obj_base) -> list:
    """
    BVH 最近面距離：找 obj_new 中靠近 obj_base 表面的頂點。
    聚類後回傳各 cluster 重心（世界座標）。
    若 min_dist > MAX_CONTACT 表示零件不相鄰，回傳 []。
    """
    bvh   = make_bvh(obj_base)
    va    = world_verts(obj_new)
    dists = np.full(len(va), np.inf, dtype=np.float64)

    for i, v in enumerate(va):
        hit = bvh.find_nearest(mathutils.Vector(v.tolist()))
        if hit is not None and hit[3] is not None:
            dists[i] = hit[3]

    finite = np.isfinite(dists)
    if not finite.any():
        return []

    min_d = dists[finite].min()
    print(f"      min_dist={min_d:.4f}", end="")

    if min_d > MAX_CONTACT:
        print(f"  > MAX_CONTACT({MAX_CONTACT}) → 跳過")
        return []

    thresh = min_d * RATIO
    close  = va[dists <= thresh]
    if len(close) == 0:
        return []

    # 貪婪 cluster（依 z→x→y 排序）
    close = close[np.lexsort((close[:, 1], close[:, 0], close[:, 2]))]
    clusters, cur = [], [close[0]]
    for v in close[1:]:
        if np.linalg.norm(v - cur[-1]) > CLUSTER_GAP:
            clusters.append(np.array(cur))
            cur = []
        cur.append(v)
    clusters.append(np.array(cur))

    print(f"  thresh={thresh:.4f}  pts={len(close)}  clusters={len(clusters)}")
    return [c.mean(axis=0).tolist() for c in clusters]


def bl_to_obj(pt_bl: list) -> list:
    """Blender world → OBJ raw: (x_bl, z_bl, -y_bl)"""
    return [pt_bl[0], pt_bl[2], -pt_bl[1]]


def project(world_pt: list, pose: dict):
    """世界座標 → 螢幕座標（Blender -Z 相機）。"""
    R_c2w   = np.array(pose["R_cam_to_world"])
    cam_loc = np.array(pose["camera_location"])
    intr    = pose["intrinsics"]
    fx, fy  = intr["fx"], intr["fy"]
    cx, cy  = intr["cx"], intr["cy"]
    W, H    = intr["width"], intr["height"]

    R_w2c = R_c2w.T
    p_cam = R_w2c @ np.array(world_pt) - R_w2c @ cam_loc
    depth = -p_cam[2]
    if depth <= 0:
        return 0.0, 0.0, False
    px = fx * p_cam[0] / depth + cx
    py = fy * (-p_cam[1]) / depth + cy
    return float(px), float(py), bool(0 <= px <= W and 0 <= py <= H)


def place_empty(name: str, loc: list):
    """在接觸點放置 Sphere Empty，加入 InstallPoints 集合。"""
    bpy.ops.object.empty_add(type='SPHERE', location=tuple(loc))
    emp = bpy.context.active_object
    emp.name = name
    emp.empty_display_size = 0.02
    for c in list(emp.users_collection):
        c.objects.unlink(emp)
    ip_coll.objects.link(emp)


def save_debug(step: str, points: list):
    """用 cv2 把接觸點圓圈畫在最後一幀渲染圖上。"""
    try:
        import cv2
    except ImportError:
        print("  ⚠️  cv2 不可用，跳過 debug 圖（可在 Blender 外執行 gen_install_points.py）")
        return
    d    = os.path.join(RENDER_DIR, step)
    pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if not pngs:
        return
    img = cv2.imread(os.path.join(d, pngs[-1]))
    for pt in points:
        if pt["in_frame"]:
            x, y = int(pt["screen_x"]), int(pt["screen_y"])
            cv2.circle(img, (x, y), 20, (0, 0, 255), 3)
            cv2.circle(img, (x, y),  3, (0, 0, 255), -1)
    out = os.path.join(DEBUG_DIR, f"{step}_gpt_check.png")
    cv2.imwrite(out, img)
    print(f"  🖼  {out}")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

for step in STEPS:
    cfg        = cfgs[step]
    show_parts = cfg["show_parts"]

    step_idx   = all_step_names.index(step)
    prev_show  = cfgs[all_step_names[step_idx - 1]]["show_parts"] if step_idx > 0 else []
    new_parts  = [p for p in show_parts if p not in prev_show]
    base_parts = [p for p in show_parts if p in prev_show]

    print(f"\n{'='*60}")
    print(f"[{step}]  new={new_parts}  base={base_parts}")
    print('='*60)

    # ── 讀相機外參
    pose = json.load(open(os.path.join(PNP_DIR, f"camera_pose_{step}.json")))
    W, H = pose["intrinsics"]["width"], pose["intrinsics"]["height"]
    print(f"  相機：{W}×{H}")

    install_points: list = []
    seen_3d: list = []

    if base_parts:
        for pair in connection_relation:
            a, b = str(pair[0]), str(pair[1])
            combos = []
            if a in new_parts and b in base_parts:
                combos.append((a, b))
            if b in new_parts and a in base_parts:
                combos.append((b, a))

            for new_pid, base_pid in combos:
                if new_pid not in parts_objs or base_pid not in parts_objs:
                    continue
                print(f"  🔍 Part{new_pid} ↔ Part{base_pid}", end="  ")
                centroids = contact_centroids_bvh(parts_objs[new_pid], parts_objs[base_pid])

                for pt3d_bl in centroids:
                    arr = np.array(pt3d_bl)
                    if any(np.linalg.norm(arr - np.array(s)) < DEDUP_R for s in seen_3d):
                        continue
                    seen_3d.append(pt3d_bl)
                    pt3d = bl_to_obj(pt3d_bl)
                    px, py, in_frame = project(pt3d, pose)
                    entry = {
                        "desc":     f"Part{new_pid}↔Part{base_pid}",
                        "world":    [round(x, 4) for x in pt3d],
                        "screen_x": round(px, 1),
                        "screen_y": round(py, 1),
                        "in_frame": in_frame,
                    }
                    install_points.append(entry)
                    tag = "✔" if in_frame else "✘"
                    print(f"  {tag} Part{new_pid}↔Part{base_pid}"
                          f"  world=({pt3d[0]:.3f},{pt3d[1]:.3f},{pt3d[2]:.3f})"
                          f"  → ({px:.0f},{py:.0f})  in_frame={in_frame}")
                    place_empty(f"IP_{step}_{new_pid}x{base_pid}", pt3d_bl)
    else:
        # base_parts 為空：往前看下一步的 new_parts，找它們與當前零件的接觸點
        next_idx = step_idx + 1
        if next_idx < len(all_step_names):
            next_show  = cfgs[all_step_names[next_idx]]["show_parts"]
            next_new   = [p for p in next_show if p not in show_parts]
            print(f"  ↪ 無 base，改看下一步 new_parts={next_new} 與 show_parts={show_parts} 接觸")
            for pair in connection_relation:
                a, b = str(pair[0]), str(pair[1])
                combos = []
                if a in next_new and b in show_parts:
                    combos.append((a, b))
                if b in next_new and a in show_parts:
                    combos.append((b, a))
                for next_pid, cur_pid in combos:
                    if next_pid not in parts_objs or cur_pid not in parts_objs:
                        continue
                    print(f"  🔍 Part{next_pid} ↔ Part{cur_pid}", end="  ")
                    centroids = contact_centroids_bvh(parts_objs[next_pid], parts_objs[cur_pid])
                    for pt3d_bl in centroids:
                        arr = np.array(pt3d_bl)
                        if any(np.linalg.norm(arr - np.array(s)) < DEDUP_R for s in seen_3d):
                            continue
                        seen_3d.append(pt3d_bl)
                        pt3d = bl_to_obj(pt3d_bl)
                        px, py, in_frame = project(pt3d, pose)
                        entry = {
                            "desc":     f"Part{next_pid}↔Part{cur_pid}",
                            "world":    [round(x, 4) for x in pt3d],
                            "screen_x": round(px, 1),
                            "screen_y": round(py, 1),
                            "in_frame": in_frame,
                        }
                        install_points.append(entry)
                        tag = "✔" if in_frame else "✘"
                        print(f"  {tag} Part{next_pid}↔Part{cur_pid}"
                              f"  world=({pt3d[0]:.3f},{pt3d[1]:.3f},{pt3d[2]:.3f})"
                              f"  → ({px:.0f},{py:.0f})  in_frame={in_frame}")
                        place_empty(f"IP_{step}_{next_pid}x{cur_pid}", pt3d_bl)

    # ── 從 small_parts.json 讀取 pad 目標數量，若 BVH 不足則補中點
    pad_counts = [e["count"] for e in small_parts_data.get(step, []) if e.get("type") == "pad"]
    pad_target = max(pad_counts) if pad_counts else 0
    if pad_target > len(install_points) and install_points:
        corners = [p["world"] for p in install_points]
        extra   = pad_target - len(corners)
        print(f"  ℹ️  pad_target={pad_target}  BVH找到={len(corners)}  補{extra}個中點")

        # 依 x 正負分組，每組內各角落平均 → 長邊中點（z 不 hardcode）
        sides: dict = {}
        for c in corners:
            sx = int(np.sign(c[0]))
            sides.setdefault(sx, []).append(c)

        added = 0
        for sx in sorted(sides.keys(), reverse=True):   # 先正後負
            if added >= extra:
                break
            pts = sides[sx]
            mid_obj = [
                round(float(np.mean([c[0] for c in pts])), 4),
                round(float(np.mean([c[1] for c in pts])), 4),
                round(float(np.mean([c[2] for c in pts])), 4),
            ]
            px, py, in_frame = project(mid_obj, pose)
            entry = {
                "desc":     f"橡膠墊中點({'+'if sx>0 else'-'}x)",
                "world":    mid_obj,
                "screen_x": round(px, 1),
                "screen_y": round(py, 1),
                "in_frame": in_frame,
            }
            install_points.append(entry)
            tag = "✔" if in_frame else "✘"
            print(f"  {tag} {entry['desc']}"
                  f"  world=({mid_obj[0]:.3f},{mid_obj[1]:.3f},{mid_obj[2]:.3f})"
                  f"  → ({px:.0f},{py:.0f})  in_frame={in_frame}")
            mid_bl = [mid_obj[0], -mid_obj[2], mid_obj[1]]
            place_empty(f"IP_{step}_mid_{'+'if sx>0 else'-'}x", mid_bl)
            added += 1

    # ── 儲存 JSON
    out_path = os.path.join(PNP_DIR, f"install_points_{step}.json")
    json.dump(install_points, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  💾 {out_path}  ({len(install_points)} 個點，"
          f"{sum(1 for p in install_points if p['in_frame'])} 個在畫面內)")

    save_debug(step, install_points)

# ── 儲存含 Empty 標記的 Blender 場景
blend_out = os.path.join(BASE_DIR, "contact_points.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
print(f"\n💾 Blender 場景：{blend_out}")

print("\n✅ 全部完成")
