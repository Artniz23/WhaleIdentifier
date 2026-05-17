from pathlib import Path
import numpy as np

def main():
    train_dir = Path("final_reid_dataset")

    whale_ids = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    np.save("individual_id.npy", np.array(whale_ids, dtype=object))

    print(f"Сохранено {len(whale_ids)} whale ids в individual_id.npy")
    print(whale_ids[:10])

if __name__ == "__main__":
    main()