# pip install -U datasets

from datasets import load_dataset, Dataset
from PIL import Image
from tqdm.auto import tqdm
DATASET_ID = "mjuicem/MM-HELIX-100K"
SPLIT = "train"
SEED = 42      # controls deterministic sampling
MAX_SIZE = 512  # max dimension for resizing images
ds = load_dataset(DATASET_ID, split=SPLIT)


selected_categories = [
    # '24Points',
#  'WordLadder', 
 'shortest_distance_weighted',
    
 ]
ds_sub = ds.filter(lambda x: x['category'] in selected_categories, num_proc=2)
print(len(ds_sub))

def resize_img(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:    # already small enough
        return img
    print(f"Resizing image from ({w}, {h}) to fit within {max_size}x{max_size}")
    scale = max_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)
    

category2metric = {'24Points': '24points_evaluator', 'aquarium': 'aquarium_evaluator', 'BestTimeToBuyAndSellStock': 'simple_str_match', 'binairo': 'binairo_evaluator', 'bridges': 'bridges_evaluator', 'Calcudoku': 'calcudoku_evaluator', 'campsite': 'campsite_evaluator', 'ContainerWithMostWater': 'simple_str_match', 'CountHillsAndValleys': 'simple_str_match', 'CryptoMath': 'cryptomath_evaluator', 'eulerian_cycle': 'eulerian_cycle_evaluator', 'eulerian_path': 'eulerian_path_evaluator', 'eulero': 'eulero_evaluator', 'futoshiki': 'futoshiki_evaluator', 'graph_isomorphism': 'simple_str_match', 'HIndex': 'simple_str_match', 'hamiltonian_cycle': 'hamiltonian_cycle_evaluator', 'hamiltonian_path': 'hamiltonian_path_evaluator', 'hitori': 'hitori_evaluator', 'kakuro': 'kakuro_evaluator', 'Kukurasu': 'kukurasu_evaluator', 'LargestRectangleInHistogram': 'simple_str_match', 'longest_increasing_subsequence': 'simple_str_match', 'max_flow': 'simple_str_match', 'maze': 'maze_evaluator', 'minesweeper': 'minesweeper_evaluator', 'nibbles': 'nibbles_evaluator', 'nonogram': 'nonogram_evaluator', 'numbrix': 'numbrix_evaluator', 'shingoki': 'shingoki_evaluator', 'shortest_distance_weighted': 'simple_str_match', 'Skyscrapers': 'skyscrapers_evaluator', 'slidingpuzzle': 'sliding_puzzle_evaluator', 'snake': 'snake_evaluator', 'sokoban': 'sokoban_evaluator', 'sudoku': 'sudoku_evaluator', 'tapa': 'tapa_evaluator', 'topological_sort': 'topological_sort_evaluator', 'hanoi': 'hanoi_evaluator', 'trapping_rain_water': 'simple_str_match', 'WordLadder': 'wordladder_evaluator', 'wordsearch': 'wordsearch_evaluator'}
rows = []
for i, datum in tqdm(enumerate(ds_sub), total=len(ds_sub)):
    # print("_"*20)
    # print("datum", datum)
    extra_info = dict(task=datum["task"], level=datum["level"], category=datum["category"], metric=category2metric[datum["category"]])
    rows.append({
        "id": i,
        "images": [resize_img(datum["image"], MAX_SIZE)],  # PIL image or path
        "question" : "<image>" + datum["question"].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'").strip(),
        "ground_truth": datum["answer"],
        "problem": "<image>" + datum["question"].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'").strip(),
        "answer": datum["cot"].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'").strip(),
        "difficulty": datum["level"],   
        "category": datum["category"],
        "initial_state": datum["initial_state"],
        "score_function": category2metric[datum["category"]],
        "revised_answer": datum["Qwen3_235B"].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'").strip(),
        "data_source": f"MM-HELIX-{','.join(selected_categories)}-{MAX_SIZE}"
    })
    # print(rows[-1])

processed_ds = Dataset.from_list(rows)
processed_ds.to_parquet(f"data/MMHELIX/MM-HELIX-{','.join(selected_categories)}-{MAX_SIZE}.parquet")
