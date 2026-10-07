from huggingface_hub import create_repo

from datasets import load_dataset, Dataset, DatasetDict
dataset = load_dataset("AI4Math/MathVista")["testmini"]

rows = []
for datum in dataset:
    print("_"*20)
    print("datum", datum)
    answer = datum["answer"]
    if datum["question_type"]=="multi_choice":
        assert answer in datum["choices"]
        response_index = datum["choices"].index(answer)
        answer = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][response_index]
    rows.append({
        "id": datum["pid"],
        "images": [datum["decoded_image"]],  # PIL image or path
        "question" : "<image>" + datum["question"],
        "ground_truth": answer,
        "problem": "<image>" + datum["query"].replace("Hint:","").strip(),
        "answer": "This is a dummy answer",
        "data_source": "mathvista"
    })
    print(rows[-1])

ds = Dataset.from_list(rows)
ds.to_parquet("data/mathvista_mini.parquet")