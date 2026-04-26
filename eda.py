import json
from tqdm import tqdm

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

def write_jsonl(path, items):
    with open(path, 'w') as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

data = read_jsonl('data/Ultrachat200k/ite0/WSPIN/train.jsonl')

print(f'keys = {data[0].keys()}')
id = 0
data_list = []

# for i in tqdm(range(len(df['real']))):
#     data_list.append({'instruction': '', 
#                       'prompt': df['real'][i][0]['content'], 
#                       'input': '',
#                       'output': df['real'][i][1]['content']})

for i in range(len(data)):
    if (len(data[i]['prompt']) > 0) and (len(data[i]['chosen']) > 0) and (len(data[i]['rejected']) > 0):
        data_list.append({'prompt': data[i]['prompt'],
                        'chosen': data[i]['chosen'],
                        'rejected': data[i]['rejected']})

output_path = "data/Ultrachat200k/ite0/WSPIN/0_true_train.jsonl"
with open(output_path, "w", encoding="utf-8") as file:
    for item in data_list:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done with {len(data_list)} sample in {output_path} while before is {len(data)}")

