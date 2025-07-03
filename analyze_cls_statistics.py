import argparse
import pickle
from collections import defaultdict
import numpy as np

def analyze_cls_statistics(cls_path):
    # === 加载数据 ===
    with open(cls_path, "rb") as f:
        data = pickle.load(f)
    print(f"✅ Loaded {len(data)} CLS entries from {cls_path}")

    # === 组织数据结构： person_id -> exp_id -> [CLS...]
    person_scene_cls = defaultdict(lambda: defaultdict(list))

    for entry in data:
        person_id = entry["person_id"]
        exp_id = entry["exp_id"]
        cls_vector = entry["cls"]  # 假定为 tensor 或 ndarray
        person_scene_cls[person_id][exp_id].append(cls_vector)

    # === 统计 ===
    person_stats = {}
    for person_id, scenes in person_scene_cls.items():
        num_scenes = len(scenes)
        cls_counts = [len(clss) for clss in scenes.values()]
        avg_cls_per_scene = np.mean(cls_counts)
        person_stats[person_id] = {
            "num_scenes": num_scenes,
            "avg_cls_per_scene": avg_cls_per_scene,
            "total_cls": sum(cls_counts)
        }

    # === 打印总体统计 ===
    print(f"\n👥 Total persons: {len(person_stats)}")
    total_scenes = sum(p["num_scenes"] for p in person_stats.values())
    print(f"📌 Total scenes: {total_scenes}")
    total_cls = sum(p["total_cls"] for p in person_stats.values())
    print(f"🔹 Total CLS vectors: {total_cls}")

    # === 打印每个 person 的统计 ===
    for person_id, stats in person_stats.items():
        print(f"Person {person_id}: Scenes={stats['num_scenes']}, "
              f"Avg CLS per scene={stats['avg_cls_per_scene']:.2f}, "
              f"Total CLS={stats['total_cls']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CLS statistics per person and scene.")
    parser.add_argument("--cls_path", type=str, required=True, help="Path to saved CLS .pkl file")
    args = parser.parse_args(
        [
            '--cls_path', '../dino_data/output_dino/cls_features.pkl'
        ]
    )

    analyze_cls_statistics(args.cls_path)
