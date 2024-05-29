# %%
from vllm import SamplingParams
import string
from vllm import LLM
from datasets import load_dataset
from tqdm import tqdm
import random
from datetime import datetime
import json
import os

# バッチサイズ
n_records = 100


os.system("mkdir -p out_data")
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"out_data/model_{current_time_no_symbols}.jsonl"


model_name = "microsoft/Phi-3-medium-128k-instruct"
# model_name="OrionStarAI/Orion-14B-Chat"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
          )

# %%

# ds=load_dataset("kanhatakeyama/ChatbotArenaJaMixtral8x22b", split="train")
# ds = load_dataset("wikipedia", "20220301.en", streaming=True, split="train")
# ds=load_dataset("hpprc/jawiki-wiktionary", split="train")
ds = load_dataset("hpprc/jawiki-books", split="train")


# %%
try:
    ds = ds.shuffle()
except:
    pass

# %%

inst_dict = {

    "textbook": """次のデータをもとに､論理的かつ教科書調の丁寧な日本語の文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
""",

    "conversation": """次のデータをもとに､論理的な日本語の会話文を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
""",
    "logical": """次のデータをもとに､論理的な文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
"""
}


def extract_random_part(text):
    text_length = len(text)
    extract_length = min(text_length, random.randint(400, 2000))
    start_index = random.randint(0, text_length - extract_length)
    return text[start_index:start_index + extract_length]


mode_list = list(inst_dict.keys())


# %%


def prepare_records(ds, n_records=10):
    ds = ds.shuffle()

    records = []
    cnt = 0
    for record in ds:
        mode = random.choice(mode_list)
        inst = inst_dict[mode]
        text = record["text"]
        text = extract_random_part(text)
        text = f"""<|user|>
    {inst}{text}<|end|>
    <|assistant|>"""
        records.append(
            {"original_text": text,
             "mode": mode,
             "url": record["url"]
             }
        )
        cnt += 1
        if cnt > n_records:
            break

    return records


# %%
while True:
    records = prepare_records(ds, n_records)
    prompts = [record["original_text"] for record in records]
    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=1024,
            repetition_penalty=1.2,
        )
    )

    for record, output in zip(records, outputs):
        record["output_text"] = (output.outputs[0].text).strip()
        record.pop("original_text")
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# %%
