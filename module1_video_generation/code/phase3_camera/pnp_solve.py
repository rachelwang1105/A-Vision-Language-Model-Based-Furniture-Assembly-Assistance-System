"""
Phase 3 - PnP Solver（普通 Python 執行）
輸入：output/pnp/points_2d_{STEP}.json + points_3d_{STEP}.json
輸出：output/pnp/camera_pose_{STEP}.json + reprojection_{STEP}.png
"""
import json, os, platform
import numpy as np
import cv2

BASE_DIR  = os.path.expanduser("~/Documents/Project/project") if platform.system() == "Darwin" else r"C:\Project"
PNP_DIR   = os.path.join(BASE_DIR, "output", "pnp")
IMG_DIR   = os.path.join(BASE_DIR, "image", "steps_crop", "vittsjo_2")
STEP      = "step04"   # ← 2D 點的步驟（說明書圖片 & 輸出檔名）
STEP_3D   = "step04"   # ← 3D 點的步驟（填同 STEP 或改用其他步驟）
TAG       = ""      # ← 同 pick_3d_points.py 的 TAG："" 或 "_ex"

# ── 載入點 ──────────────────────────────────────────────────────────
d2     = json.load(open(os.path.join(PNP_DIR, f"points_2d_{STEP}.json")))
d3     = json.load(open(os.path.join(PNP_DIR, f"points_3d_{STEP_3D}{TAG}.json")))
pts_2d = np.array(d2["points_2d"], dtype=np.float64)
pts_3d = np.array(d3["points_3d"], dtype=np.float64)
assert len(pts_2d) == len(pts_3d), f"點數不一致！2D={len(pts_2d)}, 3D={len(pts_3d)}"
print(f"載入 {len(pts_2d)} 對對應點")

# ── 圖片尺寸 & 內參 ──────────────────────────────────────────────────
img    = cv2.imread(os.path.join(IMG_DIR, f"{STEP}.png"))   # 2D 對應的說明書圖
h, w   = img.shape[:2]
focal  = float(w) * 2.5   # IKEA 說明書接近長焦，約 22° 水平 FOV
K      = np.array([[focal, 0, w/2.0],
                   [0, focal, h/2.0],
                   [0,     0,   1.0]], dtype=np.float64)
dist   = np.zeros((4, 1))
print(f"圖片：{w}×{h}，fx={focal:.0f}（水平 FOV {2*np.degrees(np.arctan(w/(2*focal))):.1f}°）")

# ── PnP 求解（自動嘗試多種方法，選 3D 點在相機前方的解） ────────────
R = rvec = tvec = None
for flag_name, flag in [("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
                        ("EPNP",      cv2.SOLVEPNP_EPNP),
                        ("SQPNP",     cv2.SOLVEPNP_SQPNP)]:
    ok, rv, tv = cv2.solvePnP(pts_3d, pts_2d, K, dist, flags=flag)
    if not ok:
        continue
    R_try, _ = cv2.Rodrigues(rv)
    pts_cam_z = (R_try @ pts_3d.T + tv)[2]   # Z in camera space
    print(f"  {flag_name}: 相機前方 Z 平均={pts_cam_z.mean():.3f} {'✓' if pts_cam_z.mean()>0 else '✗ 點在相機後方'}")
    if pts_cam_z.mean() > 0 and R is None:
        R, rvec, tvec = R_try, rv, tv         # 取第一個 Z>0 的解

if R is None:
    print("❌ 所有方法的相機都朝反向——3D 點對應位置可能選錯（選到背面的頂點），請重新標點")
    exit(1)

# ── 轉換為 Blender 相機格式 ──────────────────────────────────────────
# OpenCV cam：X 右、Y 下、Z 前 → Blender cam：X 右、Y 上、Z 後
flip           = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
cam_pos        = (-R.T @ tvec).flatten()
R_cam_to_world = R.T @ flip

# ── 重投影誤差 ────────────────────────────────────────────────────────
reproj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
reproj    = reproj.reshape(-1, 2)
errors    = np.linalg.norm(pts_2d - reproj, axis=1)
mean_err  = float(errors.mean())

print(f"\n相機位置：({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")
print(f"重投影平均誤差：{mean_err:.2f} px")
for i, (p, r, e) in enumerate(zip(pts_2d, reproj, errors)):
    print(f"  點{i}: 原{tuple(p.astype(int))} → ({r[0]:.1f},{r[1]:.1f})  {e:.1f}px")

# ── 視覺化 ────────────────────────────────────────────────────────────
vis = img.copy()
for i, (p, r) in enumerate(zip(pts_2d, reproj)):
    cv2.circle(vis, tuple(p.astype(int)), 7, (0,255,0), 2)
    cv2.circle(vis, (int(r[0]),int(r[1])), 7, (0,0,255), 2)
    cv2.line(vis, tuple(p.astype(int)), (int(r[0]),int(r[1])), (255,200,0), 1)
    cv2.putText(vis, str(i), (int(p[0])+9,int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
cv2.imwrite(os.path.join(PNP_DIR, f"reprojection_{STEP}.png"), vis)

# ── 儲存 ─────────────────────────────────────────────────────────────
focal_mm = 36.0 * focal / w   # Blender 焦距（mm），36mm 感光元件
result = {
    "step": STEP,
    "intrinsics": {"fx": focal, "fy": focal, "cx": w/2.0, "cy": h/2.0,
                   "width": w, "height": h, "focal_mm_blender": focal_mm},
    "camera_location":    cam_pos.tolist(),
    "R_cam_to_world":     R_cam_to_world.tolist(),
    "reprojection_error_mean_px": mean_err,
}
json.dump(result, open(os.path.join(PNP_DIR, f"camera_pose_{STEP_3D}{TAG}.json"), "w"), indent=2)
print(f"\n💾 camera_pose_{STEP_3D}{TAG}.json  →  執行 pnp_apply.py 套用到 Blender")
