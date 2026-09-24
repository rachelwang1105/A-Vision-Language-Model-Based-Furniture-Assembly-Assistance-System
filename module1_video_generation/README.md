# Module 1 — Assembly Manual to Video Generation

將家具組裝說明書(截圖)自動轉換成帶語音旁白的 3D 動畫解說影片。以 IKEA VITTSJO_2 玻璃桌為範例,整體流程分成三個階段:AI 視覺分析 → Blender 3D 組裝動畫 → 相機標定與影片後製合成。

## 環境需求

- 全流程皆在 **Mac** 上執行(Phase 1 AI 分析、Phase 2 Blender 動畫渲染、Phase 3 相機標定 / 後製)
- Python 3.x + [Blender](https://www.blender.org/)(建議 App 內建路徑 `/Applications/Blender.app/Contents/MacOS/Blender`)
- 安裝依賴:`pip install -r requirements.txt`
- 根目錄需要 `.env` 檔設定 `OPENAI_API_KEY`(GPT-4o 視覺分析用,不上傳版控)

> 腳本內路徑目前是寫死的:`BASE_DIR = ~/Documents/Project/project`。執行前請將本資料夾內容放到對應路徑,或修改各腳本開頭的 `BASE_DIR`。

## 資料夾結構

```
module1_video_generation/
├── code/
│   ├── phase1_analyze/       # AI 視覺分析
│   ├── phase2_align/         # Blender 組裝動畫
│   └── phase3_camera/        # 相機標定與影片後製
├── icons/                    # 零件圖示（gen_small_parts.py 自動產生）
├── image/steps_crop/vittsjo_2/   # 說明書步驟截圖（輸入來源）
├── video.blend                # Blender 主場景檔
├── requirements.txt
└── README.md
```

## 執行流程

### Phase 1:說明書圖片 AI 分析

```bash
python3 code/phase1_analyze/gen_small_parts.py       # 產出零件清單 + 圖示
python3 code/phase1_analyze/gen_narration.py         # 產出各步驟語音旁白文字
```

`gen_contact_points_blender.py` 需在 Blender 內執行,用 BVH 找零件接觸點,需先完成 Phase 3 的相機標定(見下方):

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background --python code/phase1_analyze/gen_contact_points_blender.py
```

### Phase 3(前段):相機標定(PnP)

在渲染動畫前,需先標定各步驟相機姿態,讓 3D 畫面貼合說明書視角:

1. `code/phase3_camera/pick_2d_points.py` — 在說明書截圖上點選特徵點
2. `code/phase3_camera/pick_3d_points.py` — 在 Blender Scripting 面板選取對應 3D 頂點(`MODE="record"/"save"` 等)
3. `code/phase3_camera/pnp_solve.py` — 用 OpenCV solvePnP 反推相機姿態,輸出 `output/pnp/camera_pose_stepXX.json`
4. `code/phase3_camera/pnp_apply.py` — 在 Blender 內套用相機姿態驗證效果
5. `code/phase3_camera/pnp_load_ex.py` — 標定 step05(完成組裝)所需的 `_ex` 版本流程

### Phase 2:Blender 組裝動畫渲染

相機標定完成後,批次渲染四個步驟的組裝動畫:

```bash
python3 code/phase3_camera/render_all.py
```

此腳本會依序背景呼叫 Blender 執行 `code/phase2_align/step_anim.py`,依 `code/phase2_align/step/stepXX.json` 的設定(顯示哪些零件、沿連接關係計算飛入方向)渲染出 PNG 序列到 `output/renders/stepXX/`。

### Phase 3(後段):影片後製合成

```bash
python3 code/phase3_camera/make_video.py
```

疊加說明書細節放大圖、零件圖示面板、安裝點標記,並用 edge-tts 生成語音、ffmpeg 合成最終影片,輸出到 `output/final/stepXX_final.mp4`。

## 核心資料檔案

| 檔案 | 說明 |
|---|---|
| `dataset/main_data.json`(不隨此模組上傳) | 零件 3D 模型的連接關係(`connection_relation`),用於推算飛入方向；**不含**零件擺放矩陣,零件位置由 OBJ 檔本身座標決定 |
| `output/pnp/camera_pose_stepXX(_ex).json` | Phase 3 標定出的相機姿態,渲染與後製共用 |
| `output/pnp/install_points_stepXX.json` | 零件接觸點,後製時疊加安裝點標記用 |
| `dataset/small_parts.json` | 各步驟零件清單與數量,後製疊加零件面板用 |
