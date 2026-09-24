"""
椅子組裝步驟辨識 - Prompt 設計比較測試腳本

比較「純圖片」(image_only) 與「結構化文字描述 + 照片」(structured) 兩種
prompt 設計方式，對步驟辨識準確率與延遲的影響。
"""

import base64
import json
import os
import platform
import re
import time
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 路徑設定
# ---------------------------------------------------------------------------

if platform.system() == "Darwin":
    BASE_DIR = os.path.expanduser("~/Documents/Project/ step＿recognition")
else:
    BASE_DIR = r"C:\Project"

MANUAL_IMAGE = os.path.join(BASE_DIR, "manual", "steps.jpg")
PARTS_IMAGE = os.path.join(BASE_DIR, "manual", "components.jpg")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
OUTPUT_JSON = os.path.join(BASE_DIR, "comparison_results.json")

MODEL = "gpt-5.4-mini"
SLEEP_BETWEEN_CALLS = 0.5
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
REPEATS = 3  # 每張圖每個版本重跑幾次，取多數決降低 LLM 輸出的隨機性

# ---------------------------------------------------------------------------
# 步驟定義
# ---------------------------------------------------------------------------

UNCLEAR_INSTRUCTION = """
如果照片角度、距離、光線不足或零件被遮擋，導致你無法確定目前完成到哪一步，不要用猜的：
current_step 請回答 "unclear"，reason 說明看不清楚的原因（例如：只拍到座面局部、角度看不到零件、畫面模糊），suggestion 具體告訴使用者該怎麼重新拍攝才能判斷（例如：請改用側面或斜角拍攝、請靠近拍攝座面底部中央、請確保光線充足）。"""

SELFCHECK_INSTRUCTION = """
自我檢查（很重要）：寫完 reason 之後，重新讀一次你剛剛寫的內容，確認 reason 裡描述你實際觀察到的零件狀態，跟最後選的 current_step 數字是否邏輯一致。
舉例：如果 reason 說「沒有看到椅背」，current_step 就不可以選需要椅背才成立的 Step 3 或更後面的步驟；如果 reason 說「已看到升降零件、但沒有看到椅背」，current_step 就必須是 Step 2，不能是 Step 1 或 Step 3。
一定要以 reason 裡實際描述的觀察內容為準來決定 current_step，兩者不可以互相矛盾。"""

IMAGE_ONLY_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）。
注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 消融重測用：跟 IMAGE_ONLY_PROMPT 內容完全一樣，只把 JSON 欄位順序改成 reason 在前
IMAGE_ONLY_REORDERED_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）。
注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
請先完整寫出 reason（你的觀察與推理過程），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察與推理過程", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

STRUCTURED_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

零件外觀辨識重點（step02 和 step03 容易搞混，特別注意）：
- 升降零件：座面底部中央只有一個「金屬圓形底座＋黑色旋鈕狀機構」，形狀類似軸承或關節，沒有任何布料或板狀物立起來。
- 椅背：座面後方多了一大片黑色網布（mesh）靠背板，頂端是弧形框邊，中間有金屬管骨架插入座面連接座；重點是「有沒有一整片網布板立在座面後方」，不是只看中央那個連接座長什麼樣（兩者中央連接座外觀很像，容易誤判）。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 座面後方有一整片網布椅背立起來 → step03
4. 座面中央有連接座/旋鈕機構，但沒有網布椅背 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 消融測試用：跟 STRUCTURED_PROMPT 內容完全一樣（細節導向、沒有輪廓判斷法），只把 JSON 欄位順序改成 reason 在前，
# 用來單獨驗證「欄位順序」本身的貢獻，跟輪廓判斷法的貢獻分開看
STRUCTURED_DETAIL_REORDERED_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

零件外觀辨識重點（step02 和 step03 容易搞混，特別注意）：
- 升降零件：座面底部中央只有一個「金屬圓形底座＋黑色旋鈕狀機構」，形狀類似軸承或關節，沒有任何布料或板狀物立起來。
- 椅背：座面後方多了一大片黑色網布（mesh）靠背板，頂端是弧形框邊，中間有金屬管骨架插入座面連接座；重點是「有沒有一整片網布板立在座面後方」，不是只看中央那個連接座長什麼樣（兩者中央連接座外觀很像，容易誤判）。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 座面後方有一整片網布椅背立起來 → step03
4. 座面中央有連接座/旋鈕機構，但沒有網布椅背 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
請先完整寫出 reason（你的觀察與推理過程），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察與推理過程", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# STRUCTURED_PROMPT 再加一段自我檢查，解決「reason 講的跟 current_step 選的互相矛盾」的問題
STRUCTURED_SELFCHECK_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

零件外觀辨識重點（step02 和 step03 容易搞混，特別注意）：
- 升降零件：座面底部中央只有一個「金屬圓形底座＋黑色旋鈕狀機構」，形狀類似軸承或關節，沒有任何布料或板狀物立起來。
- 椅背：座面後方多了一大片黑色網布（mesh）靠背板，頂端是弧形框邊，中間有金屬管骨架插入座面連接座；重點是「有沒有一整片網布板立在座面後方」，不是只看中央那個連接座長什麼樣（兩者中央連接座外觀很像，容易誤判）。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 座面後方有一整片網布椅背立起來 → step03
4. 座面中央有連接座/旋鈕機構，但沒有網布椅背 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + SELFCHECK_INSTRUCTION + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 用「整體輪廓形狀」取代「零件細節」判斷 step02/step03，人眼辨識時通常先看輪廓、不是先看細節
STRUCTURED_SILHOUETTE_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷 step02 和 step03 的方法（優先用「整體輪廓」判斷，比看零件細節更可靠）：
不用勉強看清楚座面中央零件的細節，先看整個物體的外觀輪廓：
- 如果整體輪廓只有座墊本身的形狀（一塊方形/橢圓形，沒有其他東西從座墊邊緣明顯突出去），代表還沒有椅背，最多到 Step 2。
- 如果輪廓除了座墊之外，還有一大片明顯突出、角度跟座墊不同、材質看起來是網狀/有洞（不是實心平滑布料）的板子附加在座墊後方，那就代表已經裝上椅背，至少是 Step 3。椅背通常跟座墊差不多大或更大，不是座墊上一個小凸起。
- 只有在輪廓判斷不出來時，才需要嘗試看座墊中央那個小零件（金屬圓形底座+黑色旋鈕狀機構＝升降零件，沒有的話代表還是 Step 1）。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 輪廓多一塊網布椅背立起來 → step03
4. 輪廓只有座墊本身，中央有連接座/旋鈕機構 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 在 STRUCTURED_SILHOUETTE_PROMPT 基礎上，修正「把升降零件誤判成扶手零件」+「用還缺什麼反推答案」這兩個 bug
STRUCTURED_V2_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷 step02 和 step03 的方法（優先用「整體輪廓」判斷，比看零件細節更可靠）：
不用勉強看清楚座面中央零件的細節，先看整個物體的外觀輪廓：
- 如果整體輪廓只有座墊本身的形狀（一塊方形/橢圓形，沒有其他東西從座墊邊緣明顯突出去），代表還沒有椅背，最多到 Step 2。
- 如果輪廓除了座墊之外，還有一大片明顯突出、角度跟座墊不同、材質看起來是網狀/有洞（不是實心平滑布料）的板子附加在座墊後方，那就代表已經裝上椅背，至少是 Step 3。椅背通常跟座墊差不多大或更大，不是座墊上一個小凸起。
- 只有在輪廓判斷不出來時，才需要嘗試看座墊中央那個小零件（金屬圓形底座+黑色旋鈕狀機構＝升降零件，沒有的話代表還是 Step 1）。

重要澄清（不要把升降零件跟扶手固定座搞混）：
座面底部通常會看到兩種不同的零件，位置和形狀都不一樣，不要混為一談：
- 扶手固定座：在座面「左右兩側」（靠近扶手那一邊），形狀扁平、常常是彎折的臂狀結構，直接連接到扶手。
- 升降零件：在座面「正中央」（左右扶手的正中間），是一個獨立突出的圓柱形旋鈕，形狀像瓶蓋或粗螺絲頭，跟兩側的扶手固定座是分開的兩個零件。
只要在座面「正中央」看到任何圓柱形/旋鈕狀的凸起物，就是升降零件，不要因為它「外觀像機構零件」就誤判成扶手零件的一部分而忽略它。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 輪廓多一塊網布椅背立起來 → step03
4. 輪廓只有座墊本身，中央有連接座/旋鈕機構 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。判斷時請先列出你在座面上「已經確認裝上」的所有零件，再從這份清單決定已完成的最高步驟，不要用「還缺什麼」來反推答案。
""" + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 精簡版：把 STRUCTURED_SILHOUETTE_PROMPT 裡「輪廓判斷法」和「判斷順序」重複講的部分合併成一份，只講一次
STRUCTURED_LEAN_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷順序（從最容易辨識開始，優先看整體輪廓，不用勉強看清楚零件細節）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 輪廓除了座墊之外，還有一大片明顯突出、材質像網狀有洞的板子（椅背，通常跟座墊差不多大或更大）→ step03
4. 輪廓只有座墊本身的形狀，沒有額外突出的大片板子，但座墊中央有金屬圓形底座+黑色旋鈕狀機構（升降零件）→ step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 在 STRUCTURED_SILHOUETTE_PROMPT 基礎上，把 JSON 欄位順序改成先寫 reason 再寫 current_step，
# 讓模型先把觀察與推理生成完，current_step 才有完整推理可以參考，避免「reason 講的跟 current_step 選的矛盾」
STRUCTURED_REORDERED_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷 step02 和 step03 的方法（優先用「整體輪廓」判斷，比看零件細節更可靠）：
不用勉強看清楚座面中央零件的細節，先看整個物體的外觀輪廓：
- 如果整體輪廓只有座墊本身的形狀（一塊方形/橢圓形，沒有其他東西從座墊邊緣明顯突出去），代表還沒有椅背，最多到 Step 2。
- 如果輪廓除了座墊之外，還有一大片明顯突出、角度跟座墊不同、材質看起來是網狀/有洞（不是實心平滑布料）的板子附加在座墊後方，那就代表已經裝上椅背，至少是 Step 3。椅背通常跟座墊差不多大或更大，不是座墊上一個小凸起。
- 只有在輪廓判斷不出來時，才需要嘗試看座墊中央那個小零件（金屬圓形底座+黑色旋鈕狀機構＝升降零件，沒有的話代表還是 Step 1）。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 輪廓多一塊網布椅背立起來 → step03
4. 輪廓只有座墊本身，中央有連接座/旋鈕機構 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
請先完整寫出 reason（你的觀察與推理過程），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察與推理過程", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 在 STRUCTURED_REORDERED_PROMPT 基礎上，把備案條款裡「金屬圓形底座+黑色旋鈕狀機構」這種具體外觀描述改模糊，
# 因為這段具體描述會誘導模型把外觀相似的扶手固定座誤判成升降零件（跟 step_defs 引發的幻覺是同一個機制）
STRUCTURED_SILHOUETTE_VAGUE_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷 step02 和 step03 的方法（優先用「整體輪廓」判斷，比看零件細節更可靠）：
不用勉強看清楚座面中央零件的細節，先看整個物體的外觀輪廓：
- 如果整體輪廓只有座墊本身的形狀（一塊方形/橢圓形，沒有其他東西從座墊邊緣明顯突出去），代表還沒有椅背，最多到 Step 2。
- 如果輪廓除了座墊之外，還有一大片明顯突出、角度跟座墊不同、材質看起來是網狀/有洞（不是實心平滑布料）的板子附加在座墊後方，那就代表已經裝上椅背，至少是 Step 3。椅背通常跟座墊差不多大或更大，不是座墊上一個小凸起。
- 只有在輪廓判斷不出來時，才需要嘗試看座墊正中央有沒有明顯的連接座/旋鈕機構（沒有的話代表還是 Step 1）。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 輪廓多一塊網布椅背立起來 → step03
4. 輪廓只有座墊本身，中央有連接座/旋鈕機構 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
請先完整寫出 reason（你的觀察與推理過程），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察與推理過程", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 在 STRUCTURED_REORDERED_PROMPT 基礎上，降低 unclear 的觸發門檻，並比照 reason 前置的邏輯，
# 要求先評估把握度再決定要不要給 unclear
STRUCTURED_CONFIDENCE_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷 step02 和 step03 的方法（優先用「整體輪廓」判斷，比看零件細節更可靠）：
不用勉強看清楚座面中央零件的細節，先看整個物體的外觀輪廓：
- 如果整體輪廓只有座墊本身的形狀（一塊方形/橢圓形，沒有其他東西從座墊邊緣明顯突出去），代表還沒有椅背，最多到 Step 2。
- 如果輪廓除了座墊之外，還有一大片明顯突出、角度跟座墊不同、材質看起來是網狀/有洞（不是實心平滑布料）的板子附加在座墊後方，那就代表已經裝上椅背，至少是 Step 3。椅背通常跟座墊差不多大或更大，不是座墊上一個小凸起。
- 只有在輪廓判斷不出來時，才需要嘗試看座墊中央那個小零件（金屬圓形底座+黑色旋鈕狀機構＝升降零件，沒有的話代表還是 Step 1）。

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 輪廓多一塊網布椅背立起來 → step03
4. 輪廓只有座墊本身，中央有連接座/旋鈕機構 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
把握度檢查（門檻降低，比之前更容易觸發 unclear）：寫完 reason 之後，問自己——如果照片角度、遮擋、模糊等因素，讓你對「有沒有看到椅背」這個關鍵判斷的把握不到七成（即使你還是傾向某個答案，只是不確定），就不要選那個數字，current_step 請回答 "unclear"。不要因為「總要選一個看起來最接近的答案」而勉強給出數字。
請先完整寫出 reason（你的觀察、推理過程、以及把握度檢查結果），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察、推理過程與把握度檢查結果", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""


# 消融實驗用：STRUCTURED_PROMPT 拆成的中間版本，用來看每個元素各自的貢獻
STEP_DEFS_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 消融重測用：跟 STEP_DEFS_PROMPT 內容完全一樣，只把 JSON 欄位順序改成 reason 在前
STEP_DEFS_REORDERED_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
請先完整寫出 reason（你的觀察與推理過程），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察與推理過程", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

CHECKLIST_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 座面後方有一整片網布椅背立起來 → step03
4. 座面中央有連接座/旋鈕機構，但沒有網布椅背 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
只回答 JSON：{"current_step": X 或 "unclear", "reason": "判斷理由或看不清楚的原因", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 消融測試用：跟 CHECKLIST_PROMPT 內容完全一樣（沒有零件外觀描述），只把 JSON 欄位順序改成 reason 在前，
# 用來驗證「reorder 之後，零件外觀描述還有沒有額外貢獻」
CHECKLIST_REORDERED_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

步驟定義：
Step 1：座面裝上左右扶手，用6顆M6×25螺絲固定
Step 2：座面裝上昇降零件，用4顆M6×20螺絲固定
Step 3：座面裝上椅背，用4顆M6×30螺絲固定
Step 4：五爪腳裝上腳輪、氣壓棒和保護套
Step 5：將座面組合插入五爪腳，完成組裝

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 座面後方有一整片網布椅背立起來 → step03
4. 座面中央有連接座/旋鈕機構，但沒有網布椅背 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
請先完整寫出 reason（你的觀察與推理過程），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察與推理過程", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""

# 消融測試用：跟 CHECKLIST_REORDERED_PROMPT 相比，拿掉「步驟定義」區塊，只留判斷順序 checklist，
# 用來驗證步驟定義這塊文字是不是也是多餘的
CHECKLIST_ONLY_PROMPT = """你是椅子組裝助理，正在辨識組裝進度。請參考說明書和零件圖，判斷實況照片目前完成到哪個步驟（1~5）

判斷順序（從最容易辨識開始）：
1. 有完整椅子（椅背+座面+扶手+五爪腳+腳輪全部組合）→ step05
2. 有五爪腳加腳輪，但座面還沒組合上來 → step04
3. 座面後方有一整片網布椅背立起來 → step03
4. 座面中央有連接座/旋鈕機構，但沒有網布椅背 → step02
5. 只有座面加左右扶手，其他都沒有 → step01

注意：判斷的是「已完成」到哪步，不是「正在做」哪步。
""" + UNCLEAR_INSTRUCTION + """
請先完整寫出 reason（你的觀察與推理過程），寫完 reason 之後，再根據 reason 裡的內容決定 current_step，確保兩者一致。
只回答 JSON，且 JSON 欄位順序必須是 reason 在前、current_step 在後：{"reason": "先寫觀察與推理過程", "current_step": X 或 "unclear", "suggestion": "current_step 為 unclear 時才需要，告訴使用者如何重拍"}"""


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def media_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    raise ValueError(f"不支援的圖片格式: {path}")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_content(path: str) -> dict:
    b64 = encode_image(path)
    media_type = media_type_for(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}"},
    }


def parse_expected_step(filename: str) -> int | None:
    match = re.match(r"step(\d+)_", filename)
    if not match:
        return None
    return int(match.group(1))


def strip_markdown_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_current_step(value):
    """模型偶爾會回傳 "step01"、"Step 3" 這種帶文字的格式，統一轉成 int 或 "unclear"。"""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.strip().lower() == "unclear":
            return "unclear"
        digits = re.search(r"\d+", value)
        if digits:
            return int(digits.group())
    return value


def parse_model_response(raw_text: str) -> dict:
    cleaned = strip_markdown_json(raw_text)
    try:
        data = json.loads(cleaned)
        return {
            "current_step": normalize_current_step(data.get("current_step")),
            "reason": data.get("reason", ""),
            "suggestion": data.get("suggestion", ""),
        }
    except json.JSONDecodeError:
        # 模型偶爾會漏掉逗號，例如 {"current_step": 3 "reason": "..."}
        # 退而求其次改用 regex 直接抓欄位，抓不到 current_step 才視為真正失敗
        step_match = re.search(r'"current_step"\s*:\s*"?(unclear|[^",}]+)"?', cleaned)
        if not step_match:
            raise
        reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        suggestion_match = re.search(r'"suggestion"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        return {
            "current_step": normalize_current_step(step_match.group(1)),
            "reason": reason_match.group(1) if reason_match else cleaned,
            "suggestion": suggestion_match.group(1) if suggestion_match else "",
        }


# ---------------------------------------------------------------------------
# API 呼叫
# ---------------------------------------------------------------------------

def call_with_prompt(client: OpenAI, prompt: str, test_image_path: str) -> dict:
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content(MANUAL_IMAGE),
                        image_content(PARTS_IMAGE),
                        image_content(test_image_path),
                    ],
                }
            ],
        )
        latency_ms = round((time.time() - start) * 1000)
        raw_text = response.choices[0].message.content
        parsed = parse_model_response(raw_text)
        parsed["latency_ms"] = latency_ms
        return parsed
    except Exception as e:
        latency_ms = round((time.time() - start) * 1000)
        return {"current_step": None, "reason": "", "suggestion": "", "error": str(e), "latency_ms": latency_ms}


def call_image_only(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, IMAGE_ONLY_PROMPT, test_image_path)


def call_image_only_reordered(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, IMAGE_ONLY_REORDERED_PROMPT, test_image_path)


def call_step_defs(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STEP_DEFS_PROMPT, test_image_path)


def call_step_defs_reordered(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STEP_DEFS_REORDERED_PROMPT, test_image_path)


def call_checklist(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, CHECKLIST_PROMPT, test_image_path)


def call_checklist_reordered(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, CHECKLIST_REORDERED_PROMPT, test_image_path)


def call_checklist_only(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, CHECKLIST_ONLY_PROMPT, test_image_path)


def call_structured(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_PROMPT, test_image_path)


def call_structured_detail_reordered(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_DETAIL_REORDERED_PROMPT, test_image_path)


def call_structured_selfcheck(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_SELFCHECK_PROMPT, test_image_path)


def call_structured_silhouette(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_SILHOUETTE_PROMPT, test_image_path)


def call_structured_v2(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_V2_PROMPT, test_image_path)


def call_structured_lean(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_LEAN_PROMPT, test_image_path)


def call_structured_reordered(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_REORDERED_PROMPT, test_image_path)


def call_structured_silhouette_vague(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_SILHOUETTE_VAGUE_PROMPT, test_image_path)


def call_structured_confidence(client: OpenAI, test_image_path: str) -> dict:
    return call_with_prompt(client, STRUCTURED_CONFIDENCE_PROMPT, test_image_path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def collect_test_images(test_dir: str) -> list[str]:
    files = [
        f
        for f in os.listdir(test_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    return sorted(files)


def run_repeated(call_fn, client: OpenAI, test_image_path: str, expected: int, repeats: int = REPEATS) -> dict:
    """對同一張圖跑 N 次，用多數決決定最終答案，同時記錄每次原始結果，方便看模型輸出的一致性。"""
    runs = []
    for _ in range(repeats):
        result = call_fn(client, test_image_path)
        result["correct"] = result.get("current_step") == expected
        runs.append(result)
        time.sleep(SLEEP_BETWEEN_CALLS)

    valid_steps = [r["current_step"] for r in runs if r.get("error") is None]
    counts = Counter(valid_steps)
    if counts:
        max_count = max(counts.values())
        top_values = [v for v, c in counts.items() if c == max_count]
        is_tied = len(top_values) > 1
        # 平手時 most_common(1) 會照第一次出現的順序挑一個，不代表真的有共識
        majority_step = counts.most_common(1)[0][0]
    else:
        is_tied = False
        majority_step = None

    correct_count = sum(1 for r in runs if r.get("correct") is True)
    latencies = [r["latency_ms"] for r in runs if "latency_ms" in r]

    return {
        "runs": runs,
        "majority_step": majority_step,
        "is_tied": is_tied,
        # 平手代表沒有真正共識，一律算錯，不讓 Counter 的任意 tiebreak 決定對錯
        "majority_correct": (not is_tied) and (majority_step == expected),
        "run_level_accuracy": round(correct_count / repeats, 4) if repeats else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
    }


def summarize(results: list[dict], key: str) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r[key]["majority_correct"] is True)
    unclear = sum(1 for r in results if r[key]["majority_step"] == "unclear")
    tied = sum(1 for r in results if r[key]["is_tied"])
    avg_latencies = [r[key]["avg_latency_ms"] for r in results]
    avg_latency = round(sum(avg_latencies) / len(avg_latencies)) if avg_latencies else 0
    accuracy = round(correct / total, 4) if total else 0.0
    run_level_accs = [r[key]["run_level_accuracy"] for r in results]
    avg_run_level_accuracy = round(sum(run_level_accs) / len(run_level_accs), 4) if run_level_accs else 0.0
    return {
        "correct": correct,
        "unclear": unclear,
        "tied": tied,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "avg_run_level_accuracy": avg_run_level_accuracy,  # 每張圖 N 次裡答對的平均比例，越接近 accuracy 代表模型越穩定
    }


def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("找不到 OPENAI_API_KEY，請確認 .env 檔案設定")
    base_url = os.environ.get("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    if not os.path.isdir(TEST_DIR):
        raise RuntimeError(f"找不到測試圖片資料夾: {TEST_DIR}")

    image_files = collect_test_images(TEST_DIR)
    if not image_files:
        raise RuntimeError(f"{TEST_DIR} 中沒有任何 jpg/png 圖片")

    print(f"共找到 {len(image_files)} 張測試圖片，每個版本每張圖重跑 {REPEATS} 次\n")

    results = []

    for idx, filename in enumerate(image_files, start=1):
        expected = parse_expected_step(filename)
        test_image_path = os.path.join(TEST_DIR, filename)

        print(f"[{idx}/{len(image_files)}] {filename} (expected=step{expected})")

        image_only_agg = run_repeated(call_image_only, client, test_image_path, expected)
        structured_agg = run_repeated(call_structured, client, test_image_path, expected)

        def fmt(agg):
            steps = [r["current_step"] if not r.get("error") else "ERR" for r in agg["runs"]]
            mark = "✓" if agg["majority_correct"] else "✗"
            tie_flag = " [TIE-無共識]" if agg["is_tied"] else ""
            return (
                f"majority={agg['majority_step']} {mark}{tie_flag} | runs={steps} | "
                f"一致率={agg['run_level_accuracy'] * 100:.0f}% | avg={agg['avg_latency_ms']}ms"
            )

        print(f"  image_only: {fmt(image_only_agg)}")
        print(f"  structured: {fmt(structured_agg)}")

        results.append(
            {
                "image": filename,
                "expected": expected,
                "image_only": image_only_agg,
                "structured": structured_agg,
            }
        )

    summary = {
        "total": len(results),
        "image_only": summarize(results, "image_only"),
        "structured": summarize(results, "structured"),
    }

    output = {"summary": summary, "results": results}

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n===== 統計摘要 =====")
    print(f"總共：{summary['total']} 張")
    io = summary["image_only"]
    st = summary["structured"]
    print(
        f"純圖片版（多數決）：{io['correct']}/{summary['total']} = "
        f"{io['accuracy'] * 100:.1f}%，單次呼叫平均正確率 {io['avg_run_level_accuracy'] * 100:.1f}%，"
        f"平均延遲 {io['avg_latency_ms']}ms，unclear {io['unclear']} 張，無共識(TIE) {io['tied']} 張"
    )
    print(
        f"結構化描述版（多數決）：{st['correct']}/{summary['total']} = "
        f"{st['accuracy'] * 100:.1f}%，單次呼叫平均正確率 {st['avg_run_level_accuracy'] * 100:.1f}%，"
        f"平均延遲 {st['avg_latency_ms']}ms，unclear {st['unclear']} 張，無共識(TIE) {st['tied']} 張"
    )
    print(f"\n結果已存至 {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
