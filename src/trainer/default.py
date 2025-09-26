from typing import Dict
import time, torch
import torch.nn as nn
from tqdm import tqdm   # thêm tqdm để làm progress bar

class DefaultTrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-4, betas: tuple = (0.9, 0.99),
                 epochs: int = 50, device: str = "cuda", ckpt_path: str = "best.pt") -> None:
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.crit = nn.L1Loss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)
        self.ckpt_path = ckpt_path
        self.best = -1e9

    def fit(self, train_loader, val_fn) -> None:
        for ep in range(1, self.epochs+1):
            self.model.train()
            t0 = time.time()
            loss_sum = 0.0

            # thêm tqdm vào vòng lặp batch
            pbar = tqdm(train_loader, desc=f"Epoch {ep:03d}", leave=False)
            for lr_img, hr_img in pbar:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                sr = self.model(lr_img)
                loss = self.crit(sr, hr_img)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                loss_sum += loss.item() * lr_img.size(0)

                # update thanh tqdm với loss trung bình hiện tại
                avg_loss = loss_sum / (len(pbar) * lr_img.size(0))
                pbar.set_postfix({"train_loss": f"{avg_loss:.4f}"})

            self.sch.step()

            # validation sau mỗi epoch
            val_metrics: Dict[str, float] = val_fn(self.model)
            psnr = val_metrics["psnr"]
            ssim = val_metrics["ssim"]

            print(f"[{ep:03d}] "
                  f"L1={loss_sum/len(train_loader.dataset):.4f} | "
                  f"Val PSNR={psnr:.3f} SSIM={ssim:.4f} | "
                  f"lr={self.sch.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

            if psnr > self.best:
                self.best = psnr
                torch.save(self.model.state_dict(), self.ckpt_path)

    def load_best(self) -> None:
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
