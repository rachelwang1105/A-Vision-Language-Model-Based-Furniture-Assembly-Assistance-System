# 基於視覺語言模型之家具組裝輔助系統

A Vision-Language Model-Based Furniture Assembly Assistance System

國立政治大學資訊科學系 畢業專題｜指導教授：廖文宏

以智慧眼鏡結合視覺語言模型輔助家具組裝：將紙本說明書自動轉換為 3D 教學動畫，並於組裝過程中辨識使用者當前所處的步驟。

**[模組一demo](https://youtu.be/HtNzd6He6GE)**

---

## 系統概觀

| 模組 | 執行時機 | 功能 |
|---|---|---|
| 模組一 | 離線預先產生 | 說明書 → 3D 教學動畫 |
| 模組二 | 即時執行 | 辨識使用者當前的組裝步驟 |

使用者透過智慧眼鏡拍攝當前組裝狀態，系統辨識所處步驟後以語音回報，並播放模組一預先生成的對應教學影片。

## Repo 結構

```
module1_video_generation/
├── code/          流程各階段腳本
├── icons/         自動生成的零件圖示
├── image/         說明書步驟圖（vittsjo_2）
└── video.blend    Blender 場景

module2_step_recognition/
├── prompts/                    消融實驗的各版本 prompt
├── results/                    測試結果原始數據
├── run_comparison.py           各 prompt 版本的批次比較
└── disagreement_analysis.py    三次呼叫分歧時的分析機制
```

各模組的詳細說明見 [模組一 README](module1_video_generation/README.md) 與 [模組二 README](module2_step_recognition/README.md)。

---

## 模組一：說明書轉 3D 教學動畫

以說明書步驟圖為輸入，輸出與說明書視角一致的 3D 教學影片，流程分為四個階段：

1. **說明書分析** — 以 VLM 萃取每步所需零件與數量，生成結構化 JSON 與口語化旁白（edge-tts 轉語音）
2. **安裝點計算** — 以 BVHTree 結合最近面幾何法，計算新零件與既有結構的接觸面中心
3. **動畫生成** — 零件採用資料集匯出時已對齊好的原生座標，依連接關係計算新零件的插入方向並以動畫飛入定位，EEVEE 渲染 PNG 序列
4. **相機標定與後製** — 手動標點後以 OpenCV solvePnP 反推相機外參並轉至 Blender 座標系，使渲染視角貼合說明書原圖；疊加細節對照框、零件圖示與安裝點標記，以 ffmpeg 合成 MP4

> 目前各階段為獨立執行，尚未串接為單一流程。

---

## 模組二：組裝步驟辨識

驗證視覺語言模型能否從使用者拍攝的影像判斷目前完成的步驟。測試對象為家用電競椅，共五個步驟、43 張實拍照片。為降低模型輸出的隨機性，每張照片重複呼叫三次並採多數決，三次分歧者一律計為錯誤。

### Prompt 消融實驗

| 版本 | 正確率 | 三次分歧次數 |
|---|---|---|
| image_only | 83.7% | 2 |
| + 步驟定義 | 83.7% | 1 |
| **+ 判斷順序 checklist** | **88.4%** | **0** |
| + 零件外觀描述 | 86.1% | 0 |
| 僅 checklist（無步驟定義） | 79.1% | 0 |

各版本完整 prompt 見 [`prompts/`](module2_step_recognition/prompts)，原始結果見 [`results/`](module2_step_recognition/results)。

### 關鍵發現：JSON 欄位順序影響判斷品質

實驗中出現逾 20 個「推理正確但答案錯誤」的案例——模型的文字推理明確描述看到某零件，最終選出的步驟編號卻與該觀察矛盾。

原因並非規則描述不清，而在於輸出欄位的生成順序：原格式為 `current_step` 在前、`reason` 在後，逐字生成的模型等於**被迫在寫出推理內容之前就先決定答案**。

改為 `reason` 在前、`current_step` 在後後，特定步驟的辨識率由 **20% 提升至 100%**，且三次呼叫完全一致。此為整個實驗中效果最顯著的單一修改，僅需更動一行程式碼。

其餘發現（prompt 複雜度上限、模型無法可靠自評信心、小樣本結論的侷限）見 [模組二 README](module2_step_recognition/README.md)。

---

## 眼鏡端整合

眼鏡端基於開源專案 [OpenGlasses](https://github.com/straff2002/OpenGlasses) 修改，使 Meta Ray-Ban 拍攝的影像能送入視覺語言模型分析。因涉及個人 SDK 憑證，該部分程式碼未一併公開。

實測端對端延遲約 4.7–5.3 秒。串流影格（約 36 KB）解析度不足以辨識細小零件，改為主動拍照（1440×1080，約 360 KB）後可正確辨識十字槽螺絲等細節。

## 技術棧

Python · Blender Python API · OpenCV · 視覺語言模型 API · edge-tts · ffmpeg · Meta Wearables SDK

## 現況

專題仍在進行中。模組一與模組二已分別完成並驗證，第三階段的眼鏡端整合進行中。

## 資料來源

3D 零件模型與組裝關係取自 [IKEA-Manual](https://cs.stanford.edu/~rcwang/projects/ikea_manual/) 資料集；`module1_video_generation/image/` 之說明書圖片為原廠說明書之裁切，僅供研究用途。
