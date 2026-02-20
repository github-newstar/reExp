import time
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
from tqdm import tqdm

@hydra.main(version_base="1.3", config_path="../src/configs", config_name="brats23_swin_unetr_cached")
def main(cfg: DictConfig):
    # Print the config we're actually using
    print(OmegaConf.to_yaml(cfg.datasets.train))
    print(OmegaConf.to_yaml(cfg.dataloader))

    print("\n--- 1. Initializing Train Dataset ---")
    t0 = time.time()
    train_dataset = instantiate(cfg.datasets.train)
    print(f"Dataset initialization took: {time.time() - t0:.2f}s")
    print(f"Dataset length: {len(train_dataset)}")

    print("\n--- 2. Initializing DataLoader ---")
    t1 = time.time()
    train_loader = instantiate(cfg.dataloader, dataset=train_dataset)
    print(f"DataLoader initialization took: {time.time() - t1:.2f}s")

    print("\n--- 3. Fetching First Batch (Worker Init + First IO) ---")
    t2 = time.time()
    loader_iter = iter(train_loader)
    print(f"Iterator creation took: {time.time() - t2:.2f}s")

    t3 = time.time()
    batch = next(loader_iter)
    t4 = time.time()
    print(f"First batch fetched in: {t4 - t3:.2f}s")
    print(f"Batch keys: {batch.keys()}")
    if 'image' in batch:
        print(f"Image shape: {batch['image'].shape}, dtype: {batch['image'].dtype}")

    print("\n--- 4. Fetching Next 5 Batches ---")
    for i in range(5):
        t_start = time.time()
        batch = next(loader_iter)
        t_end = time.time()
        print(f"Batch {i+2} fetched in: {t_end - t_start:.2f}s")

if __name__ == "__main__":
    main()
