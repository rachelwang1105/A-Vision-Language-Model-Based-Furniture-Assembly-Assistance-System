# Module 2 — Step Recognition（椅子組裝步驟辨識）

這個模組負責「給一張使用者拍的組裝過程照片，判斷目前完成到 5 個步驟中的哪一步」，是整個家具組裝輔助系統裡的辨識核心。本文件說明系統架構、prompt 設計的消融實驗結果，以及過程中最重要的一個技術發現。

---

## 系統架構

```
使用者照片
   │
   ▼
┌─────────────────────────────────────────┐
│  Prompt（checklist_reordered，見下方消融結果）│
│  + 說明書圖 + 零件圖 + 使用者照片           │
└─────────────────────────────────────────┘
   │
   ▼
GPT-5.4-mini（vision）× 3 次獨立呼叫
   │
   ▼
┌───────────────────────────┐
│ 多數決（majority vote）      │
│ 3 次答案一致 → 採用該答案     │
│ 3 次答案不一致（平手）→ 判定為需要人工確認 │
└───────────────────────────┘
   │
   ├─ 一致 ──────────────▶ 回傳 current_step + reason
   │
   └─ 不一致（平手）────────▶ 觸發「事後分歧分析」
                              把 3 次不同答案 + 理由攤給模型看，
                              請它指出最可能的混淆原因，
                              並給使用者一條具體的重拍建議
```

**每次模型呼叫回傳的 JSON 格式**（欄位順序見下方「關鍵發現」）：
```json
{
  "reason": "先寫觀察與推理過程",
  "current_step": 1~5 或 "unclear",
  "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"
}
```

$\color{blue}{\texttt{"current\_step": 1-5 or "unclear"}}$

---

## 消融實驗結果（43 張測試照片，每張重複 3 次多數決）

以下 5 個版本依序疊加 prompt 元素，測試在**完整資料集**（不是挑選過的乾淨子集）上的表現：

| 版本 | Prompt 內容 | 正確率 | step1 | step2 | step3 | step4 | step5 | 無共識(TIE) |
|---|---|---|---|---|---|---|---|---|
| `01_image_only` | 純圖片 + 一句話指示 | 83.7% (36/43) | 10/10 | 7/11 | 6/9 | 5/5 | 8/8 | 2 |
| `02_step_defs` | + 步驟定義文字 | 83.7% (36/43) | 8/10 | 7/11 | 8/9 | 5/5 | 8/8 | 1 |
| **`03_checklist`** | **+ 判斷順序 checklist** | **88.4% (38/43)** | **10/10** | 8/11 | 7/9 | 5/5 | 8/8 | **0** |
| `04_structured_silhouette` | + 輪廓判斷法（優先看整體形狀）| 86.1% (37/43) | 8/10 | 9/11 | 7/9 | 5/5 | 8/8 | 0 |
| `05_checklist_only` | 只有判斷順序，拿掉步驟定義 | 79.1% (34/43)* | — | — | — | — | — | 0 |

\* `05_checklist_only` 第一輪測試有 3 次連線失敗，已個別重打補齊後才計入此數字。

**目前採用版本：`03_checklist`**——只有步驟定義 + 判斷順序，沒有任何零件外觀描述或輪廓判斷法，是目前所有測試版本裡表現最好、也最簡單的一版。完整 prompt 內容見 [`prompts/03_checklist_BEST.txt`](prompts/03_checklist_BEST.txt)，原始測試數據見 [`results/03_checklist_BEST_43photos.json`](results/03_checklist_BEST_43photos.json)。

**三個關鍵發現：**
1. **加零件外觀描述容易讓模型「幻覺」看見不存在的零件**——`02`、`04` 版本都會把「只有座面+扶手」的照片誤判成「已裝上升降零件」，因為 prompt 一旦具體描述零件外觀，就會誘導模型主動去找，容易把外觀相似的扶手固定座誤認成別的零件。
2. **步驟定義文字單獨使用沒有幫助，但拿掉後 checklist 也會變差**（`05` 版本 79.1%）——兩者是互補關係，不是誰包含誰，個別測試會低估兩者合併的效果。
3. 剩下的錯誤全部集中在 step2⟷step3 互相誤判，step1、4、5 目前已經是滿分。

---

## 關鍵技術發現：JSON 欄位順序

在疊加各種 prompt 內容（零件描述、判斷順序、自我檢查指示……）都沒能解決「模型推理內容正確，但最終答案卻矛盾」的問題之後，最終找到的根本原因不在文字內容，而在 **JSON 輸出格式裡欄位的先後順序**。

- **問題**：原始格式是 `{"current_step": X, "reason": "..."}`，`current_step` 排在 `reason` 前面。LLM 是逐字生成的（自迴歸），這代表模型必須在**寫出推理內容之前**就先決定答案數字——等於「先射箭、再畫靶」，後面寫的 reason 只是替已經定案的答案找理由，而不是真的先思考再回答。
- **修法**：把欄位順序改成 `{"reason": "...", "current_step": X}`，強制模型先把觀察與推理寫完，再根據前面寫的內容決定最終答案。
- **效果**：在 step02/03 混淆最嚴重的 10 張照片子集上，這個改動讓正確率從 50% 提升到 70%，其中 step02 從 1/5 進步到 5/5（三次呼叫完全一致）。這是整個專題測試過的所有修改裡，效果最顯著的一次，而且只是把 JSON 兩個欄位的順序對調而已。

延伸閱讀：這個現象也呼應了 chain-of-thought 相關研究的一般性原則——要求模型「先推理、後給答案」通常比「先給答案、後補理由」表現更好，只是這次是在結構化 JSON 輸出的欄位順序這個具體、容易被忽略的細節上發現的。

---

## Demo 影片

https://youtu.be/HtNzd6He6GE

---

## 檔案結構

```
module2_step_recognition/
├── README.md                          本文件
├── run_comparison.py                  主程式：所有 prompt 版本定義、API 呼叫、多數決邏輯
├── disagreement_analysis.py           3 次答案不一致時的事後分歧分析（重拍建議）
├── requirements.txt                   Python 套件需求（openai、python-dotenv、Pillow）
├── prompts/                           5 個版本的完整 prompt 內容（純文字，方便直接閱讀比較）
│   ├── 01_image_only.txt
│   ├── 02_step_defs.txt
│   ├── 03_checklist_BEST.txt          目前採用版本
│   ├── 04_structured_silhouette.txt
│   └── 05_checklist_only.txt
└── results/                           43 張照片 × 3 次呼叫的完整原始測試數據
    ├── 01_image_only_43photos.json
    ├── 02_step_defs_43photos.json
    ├── 03_checklist_BEST_43photos.json
    ├── 04_structured_silhouette_43photos.json
    ├── 05_checklist_only_43photos.json
    └── 06_structured_silhouette_vague_supplementary.json   （補充實驗，未列入主表）
```

`prompts/*.txt` 裡的內容是直接從 `run_comparison.py` 對應的 prompt 常數（如 `CHECKLIST_REORDERED_PROMPT`）匯出的純文字版本，兩者內容一致，`.txt` 版本只是方便在 README 或簡報裡直接引用、不用開程式碼。

每份 `results/*.json` 都保留了每張照片 3 次呼叫各自的 `current_step`、`reason`、`latency_ms`，以及多數決後的 `majority_step`、`is_tied`、`majority_correct`，可用於重現上方表格數字或做進一步分析。
