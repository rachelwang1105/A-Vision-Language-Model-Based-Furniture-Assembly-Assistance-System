"""在 Blender 執行：根據 PnP 結果建立相機並套用到場景"""
import bpy, json, math, os, platform, mathutils

BASE_DIR  = os.path.expanduser("~/Documents/Project/project") if platform.system() == "Darwin" else r"C:\Project"
STEP      = "step02"   # ← 改這裡換步驟 (step01 / step02 / step03 / step04)
TAG       = "_ex"      # ← 同 pnp_solve.py 的 TAG："" 或 "_ex"
ROLL_180  = False      # ← 如果畫面上下顛倒改成 True

POSE_PATH = os.path.join(BASE_DIR, "output", "pnp", f"camera_pose_{STEP}{TAG}.json")
pose      = json.load(open(POSE_PATH))
loc       = pose["camera_location"]
R33       = pose["R_cam_to_world"]
focal_mm  = pose["intrinsics"]["focal_mm_blender"]
mean_err  = pose["reprojection_error_mean_px"]

R_mat = mathutils.Matrix([R33[0], R33[1], R33[2]])
mat4  = R_mat.to_4x4()
mat4.translation = mathutils.Vector(loc)

if ROLL_180:
    mat4 = mat4 @ mathutils.Matrix.Rotation(math.pi, 4, 'Z')
    print("roll 補正：180°")

cam_obj = bpy.data.objects.get("Camera")
if cam_obj is None:
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj  = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

cam_obj.matrix_world = mat4
cam_obj.data.lens    = focal_mm
bpy.context.scene.camera = cam_obj

print(f"✅ 相機已設定 [{STEP}]  ROLL_180={ROLL_180}")
print(f"   位置：({loc[0]:.3f}, {loc[1]:.3f}, {loc[2]:.3f})")
print(f"   焦距：{focal_mm:.1f} mm  重投影誤差：{mean_err:.2f} px")
