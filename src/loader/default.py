# src/loader/default.py
from typing import Tuple, List, Sequence, Optional, Callable
from glob import glob
import os, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .augment import build_pipeline as build_aug

class _PairDataset(Dataset):
    def __init__(self, hr_paths: Sequence[str], lr_paths: Sequence[str], aug = None):
        assert len(hr_paths) == len(lr_paths) and len(hr_paths) > 0
        self.hr_paths, self.lr_paths = list(hr_paths), list(lr_paths)
        self.aug = aug
        self.to_tensor = transforms.ToTensor()
    def __len__(self) -> int: return len(self.hr_paths)
    def __getitem__(self, i: int):
        hr = Image.open(self.hr_paths[i]).convert("RGB")
        lr = Image.open(self.lr_paths[i]).convert("RGB")
        if self.aug is not None: lr, hr = self.aug(lr, hr)
        return self.to_tensor(lr), self.to_tensor(hr)

class DefaultLoader:
    def __init__(self, hr_dir: str, lr_dir: str, split = (0.8, 0.1, 0.1),
                 batch_size = 16, num_workers = 4, seed = 42,
                 exts = (".png", ".jpg", ".jpeg"),
                 augment = None):
        self.hr_dir, self.lr_dir = hr_dir, lr_dir
        self.split = split
        self.batch_size, self.num_workers = batch_size, num_workers
        self.seed = seed; self.exts = exts
        self.augment = augment or {}

    def _list_images(self, d: str) -> List[str]:
        xs: List[str] = []
        for e in self.exts: xs += glob(os.path.join(d, f"*{e}"))
        return sorted(xs)

    def make(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        hr, lr = self._list_images(self.hr_dir), self._list_images(self.lr_dir)
        assert len(hr) == len(lr) and len(hr) > 0
        n = len(hr); random.seed(self.seed); ids = list(range(n)); random.shuffle(ids)
        s_tr, s_va, s_te = self.split
        n_te, n_va = int(n * s_te), int(n * s_va); n_tr = n - n_va - n_te
        id_tr, id_va, id_te = ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]
        pick = lambda a, I: [a[i] for i in I]

        aug_train = build_aug(self.augment.get("train", [])) if self.augment.get("train") else None
        aug_val   = build_aug(self.augment.get("val",   [])) if self.augment.get("val")   else None

        ds_tr = _PairDataset(pick(hr, id_tr), pick(lr, id_tr), aug=aug_train)
        ds_va = _PairDataset(pick(hr, id_va), pick(lr, id_va), aug=aug_val)
        ds_te = _PairDataset(pick(hr, id_te), pick(lr, id_te), aug=None)

        mk = lambda ds, bs, sh: DataLoader(ds, batch_size = bs, shuffle = sh,
                                           num_workers = self.num_workers, pin_memory = True)
        return mk(ds_tr, self.batch_size, True), mk(ds_va, self.batch_size, False), mk(ds_te, 1, False)
