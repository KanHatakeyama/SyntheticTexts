from vllm import SamplingParams, LLM

from datasets import load_dataset, concatenate_datasets
from datetime import datetime
import json
import os
from src.generator import inst_dict, prepare_records
import time
import random

wait_time = random.randint(1, 60)
#time.sleep(wait_time)



# バッチサイズ
n_records = 30


os.system("mkdir -p out_data")
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"out_data/model_{current_time_no_symbols}_openmath.jsonl"

ds_list = [
    # load_dataset("hpprc/jawiki-wiktionary", split="train"),
    load_dataset("kunishou/OpenMathInstruct-1-1.8m-ja", split="train"),
]
# 必要な 列だけを抽出して新しいリストに追加
ds_list_filtered = [
    ds.remove_columns(
        [col for col in ds.column_names if col not in ['question_ja', 'generated_solution_ja']])
    for ds in ds_list
]

# データセットを結合
ds = concatenate_datasets(ds_list_filtered)
#ds["text"]="問題. "+ds["question_ja"]+"\n 解答."+ds["generated_solution_ja"]

ds=ds.filter(lambda x: x["question_ja"] is not None and x["generated_solution_ja"] is not None)

#ds["text"] = "問題. " + ''.join(ds["question_ja"]) + "\n 解答." + ''.join(ds["generated_solution_ja"])
def add_text_column(example):
    question_text = (example["question_ja"])
    solution_text = (example["generated_solution_ja"])
    return {"text": "問題. " + question_text + "\n 解答." + solution_text}
ds = ds.map(add_text_column)
# ds = concatenate_datasets(ds_list)
model_name = "microsoft/Phi-3-medium-128k-instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
)


# %%
try:
    ds = ds.shuffle()
except:
    pass

# %%


mode_list = list(inst_dict.keys())


# %%
print(len(ds), " records")

# %%
while True:
    records = prepare_records(
        ds, mode_list, n_records=n_records, random_extract=True,db_name="kunishou/OpenMathInstruct-1-1.8m-ja")
    
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