"""
當 3 次重複判斷不一致時，額外呼叫一次，把 3 次的答案和理由攤開給模型看，
請它分析「為什麼會分歧」並給出具體重拍建議——不是要模型自我評估信心（已證實不可靠），
而是讓它比對三份既有的外部證據。
"""

import json
import re

DISAGREEMENT_PROMPT_TEMPLATE = """你是椅子組裝助理。系統對這張照片獨立判斷了 3 次，得到以下不同結果：

第1次：判斷為 Step {step1}，理由：{reason1}

第2次：判斷為 Step {step2}，理由：{reason2}

第3次：判斷為 Step {step3}，理由：{reason3}

這 3 次答案不一致，代表這張照片可能有辨識上的困難。請你重新看這張照片，對照上面 3 次理由裡的差異，找出**最主要的一個**造成混淆的原因（不用列出所有可能原因，只挑影響最大的那一個），並給使用者**一條**最關鍵、最值得優先做的重拍建議（不要列清單，一句話講清楚要做什麼）。

只回答 JSON：{{"confusion_reason": "一句話說明最主要的混淆原因", "suggestion": "一條最關鍵的重拍建議"}}"""


def analyze_disagreement(client, model, test_image_path, image_content_fn, runs: list[dict]) -> dict:
    prompt = DISAGREEMENT_PROMPT_TEMPLATE.format(
        step1=runs[0].get("current_step"), reason1=runs[0].get("reason"),
        step2=runs[1].get("current_step"), reason2=runs[1].get("reason"),
        step3=runs[2].get("current_step"), reason3=runs[2].get("reason"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content_fn(test_image_path),
                ],
            }
        ],
    )
    raw_text = response.choices[0].message.content
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)
