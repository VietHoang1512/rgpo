# pip install -U datasets

from datasets import load_dataset, Dataset
from collections import Counter, defaultdict
from PIL import Image
from tqdm.auto import tqdm
DATASET_ID = "tianhao2k/MM-HELIX"
SPLIT = "test"
MAX_SIZE = 512  # max dimension for resizing images

# Load dataset (default config)
ds = load_dataset(DATASET_ID, split=SPLIT)
print(len(ds), ds)
print("Columns:", ds.column_names)
print("Features:", ds.features)

categories = sorted(set(ds["category"]))    # fallback

print(f"\nUnique categories ({len(categories)}):")
for c in categories:
    print(" -", c)

# --- (Optional) category counts ---
counts = Counter(ds["category"])
print("\nCategory counts:")
for c, n in counts.items():
    print(f"{c}\t{n}")

def resize_img(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:    # already small enough
        return img
    print(f"Resizing image from ({w}, {h}) to fit within {max_size}x{max_size}")
    scale = max_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)

def str_to_dict(input_str):
    if isinstance(input_str, str):
        try:
            import json
            input_str = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            try:
                import ast
                input_str = ast.literal_eval(input_str)
            except (ValueError, SyntaxError):
                pass
    return input_str



selected_categories = [
    # '24Points',
#  'WordLadder', 
 'shortest_distance_weighted',
    
 ]
ds_sub = ds.filter(lambda x: x['category'] in selected_categories)

print(len(ds_sub), "examples after filtering")

GROUP_LIST = {
    'graph_problems': [
        'connectivity_test', 'eulerian_cycle', 'eulerian_path',
        'graph_isomorphism', 'hamiltonian_cycle', 'hamiltonian_path',
        'max_flow', 'shortest_distance_weighted', 'topological_sort',
    ],
    'puzzles': [
        'Calcudoku', 'Kukurasu', 'Skyscrapers', 'WordLadder', 'eulero',
        'numbrix', 'snake', 'aquarium', 'binairo', 'bridges', 'campsite',
        'futoshiki', 'hitori', 'kakuro', 'nonogram', 'shingoki', 'tapa',
        'wordsearch', 'sudoku'
    ],
    'algorithm_problems': [
        '24Points',
        'BestTimeToBuyAndSellStock',
        'ContainerWithMostWater', 'CountHillsAndValleys',
        'CryptoMath', 'HIndex', 'LargestRectangleInHistogram',
        'longest_increasing_subsequence',
        'trapping_rain_water',
    ],
    'games': [
        'sokoban', 'hanoi', 'maze', 'minesweeper', 'slidingpuzzle', 'nibbles',
    ],
}


category2metric = {'24Points': '24points_evaluator', 'aquarium': 'aquarium_evaluator', 'BestTimeToBuyAndSellStock': 'simple_str_match', 'binairo': 'binairo_evaluator', 'bridges': 'bridges_evaluator', 'Calcudoku': 'calcudoku_evaluator', 'campsite': 'campsite_evaluator', 'ContainerWithMostWater': 'simple_str_match', 'CountHillsAndValleys': 'simple_str_match', 'CryptoMath': 'cryptomath_evaluator', 'eulerian_cycle': 'eulerian_cycle_evaluator', 'eulerian_path': 'eulerian_path_evaluator', 'eulero': 'eulero_evaluator', 'futoshiki': 'futoshiki_evaluator', 'graph_isomorphism': 'simple_str_match', 'HIndex': 'simple_str_match', 'hamiltonian_cycle': 'hamiltonian_cycle_evaluator', 'hamiltonian_path': 'hamiltonian_path_evaluator', 'hitori': 'hitori_evaluator', 'kakuro': 'kakuro_evaluator', 'Kukurasu': 'kukurasu_evaluator', 'LargestRectangleInHistogram': 'simple_str_match', 'longest_increasing_subsequence': 'simple_str_match', 'max_flow': 'simple_str_match', 'maze': 'maze_evaluator', 'minesweeper': 'minesweeper_evaluator', 'nibbles': 'nibbles_evaluator', 'nonogram': 'nonogram_evaluator', 'numbrix': 'numbrix_evaluator', 'shingoki': 'shingoki_evaluator', 'shortest_distance_weighted': 'simple_str_match', 'Skyscrapers': 'skyscrapers_evaluator', 'slidingpuzzle': 'sliding_puzzle_evaluator', 'snake': 'snake_evaluator', 'sokoban': 'sokoban_evaluator', 'sudoku': 'sudoku_evaluator', 'tapa': 'tapa_evaluator', 'topological_sort': 'topological_sort_evaluator', 'hanoi': 'hanoi_evaluator', 'trapping_rain_water': 'simple_str_match', 'WordLadder': 'wordladder_evaluator', 'wordsearch': 'wordsearch_evaluator'}

rows = []
for i, datum in tqdm(enumerate(ds_sub), total=len(ds_sub)):
    # print("_"*20)
    # print("datum", datum)
    assert len(datum["images"]) == 1, f"Expected 1 image per example, but got {len(datum['images'])} for id {datum['id']}"
    assert str_to_dict(datum["metric_info"])["score_function"] == category2metric[datum["category"]], f"Metric info mismatch for category '{datum['category']}': expected '{category2metric[datum['category']]}', got '{str_to_dict(datum['metric_info'])['score_function']}'"
    rows.append({
        "id": i,
        "images": [resize_img(img, MAX_SIZE) for img in datum["images"]],  # PIL image or path
        "question" : "<image>" + datum["question"].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'").strip(),
        "ground_truth": datum["answer"],
        "problem": "<image>" + datum["question"].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'").strip(),
        "answer": "dummy answer",  # Placeholder since this is test set without COT
        "level": datum["difficulty"],
        "category": datum["category"],
        "score_function": str_to_dict(datum["metric_info"])["score_function"],
        "initial_state": datum["initial_state"],
        "data_source": datum["category"]
    })
    # break
print(rows[-1])
processed_ds = Dataset.from_list(rows)
processed_ds.to_parquet(f"data/MMHELIX/MM-HELIX-{','.join(selected_categories)}-{MAX_SIZE}-test.parquet")

