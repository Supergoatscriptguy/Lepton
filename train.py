"""LEPTON Training Script"""

import os, sys, time, argparse
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tokenizers import Tokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import model as lepton_model
from fast_dataloader import create_fast_dataloaders


def progress_bar(current, total, width=20):
    pct = current / total
    filled = int(width * pct)
    return f"|{'█' * filled}{'░' * (width - filled)}| {pct*100:5.1f}%"


def format_time(seconds):
    if seconds < 60: return f"{seconds:.0f}s"
    elif seconds < 3600: return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else: return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"


class LeptonTrainer:
    def __init__(self, model, train_loader, val_loader, tokenizer_path, lr=6e-4, weight_decay=0.1,
                 warmup_steps=500, max_steps=100000, grad_accum_steps=4, checkpoint_dir='checkpoints',
                 eval_interval=1000, save_interval=1000, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = checkpoint_dir
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        no_decay = ['bias', 'norm', 'embedding']
        params = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), fused=True)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=2000, T_mult=2)
        self.scaler = GradScaler('cuda')
        self.global_step = 0
        self.best_val_loss = float('inf')
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_step(self, batch):
        self.model.train()
        batch = batch.to(self.device, non_blocking=True)
        with autocast('cuda'):
            loss = self.model(batch, targets=batch)['loss'] / self.grad_accum_steps
        self.scaler.scale(loss).backward()
        return loss.item() * self.grad_accum_steps

    @torch.no_grad()
    def evaluate(self, num_batches=30):
        self.model.eval()
        total_loss, count = 0, 0
        for i, batch in enumerate(self.val_loader):
            if i >= num_batches: break
            with autocast('cuda'):
                total_loss += self.model(batch.to(self.device), targets=batch.to(self.device))['loss'].item()
            count += 1
        return total_loss / count if count > 0 else 0

    @torch.no_grad()
    def generate_sample(self, prompt="The meaning of life is", max_tokens=50):
        self.model.eval()
        tokens = self.tokenizer.encode(prompt).ids
        return self.tokenizer.decode(self.model.generate(tokens, max_new_tokens=max_tokens, temperature=0.8))

    def save_checkpoint(self, name='checkpoint'):
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        torch.save({
            'model': model_to_save.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), 'scaler': self.scaler.state_dict(),
            'global_step': self.global_step, 'best_val_loss': self.best_val_loss,
        }, os.path.join(self.checkpoint_dir, f'{name}.pt'))

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model.load_state_dict(ckpt['model'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        print(f"Resumed from step {self.global_step}")

    def train(self):
        model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        print(f"\nLEPTON Training | {model.count_parameters():,} params | {self.max_steps:,} steps\n")

        train_iter = iter(self.train_loader)
        tokens_processed = 0
        start_time = time.time()

        while self.global_step < self.max_steps:
            try: batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            loss = self.train_step(batch)
            tokens_processed += batch.numel()

            if (self.global_step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            self.global_step += 1
            elapsed = time.time() - start_time
            tps = tokens_processed / elapsed if elapsed > 0 else 0
            eta = (self.max_steps - self.global_step) / (self.global_step / elapsed) if self.global_step > 0 else 0

            print(f"\r{progress_bar(self.global_step, self.max_steps)} {self.global_step:,}/{self.max_steps:,} | "
                  f"Loss: {loss:.4f} | {tps/1000:.1f}k tok/s | ETA: {format_time(eta)}", end='', flush=True)

            if self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                sample = self.generate_sample("Artificial intelligence will")
                print(f"\n\n[Step {self.global_step}] Val: {val_loss:.4f}\n[Sample] {sample[:150]}...")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')
                    print("New best!")
                print()

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}')

            if self.global_step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.save_checkpoint('final')
        print(f"\n\nDone! Best val loss: {self.best_val_loss:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--tokenizer', type=str, default='pile_tokenizer.json')
    p.add_argument('--model_size', type=str, default='tiny', choices=['nano', 'micro', 'tiny', 'small', 'medium', 'large', 'xl', 'xxl'])
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--max_steps', type=int, default=50000)
    p.add_argument('--grad_accum', type=int, default=16)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    creators = {
        'nano': lepton_model.create_lepton_nano, 'micro': lepton_model.create_lepton_micro,
        'tiny': lepton_model.create_lepton_tiny, 'small': lepton_model.create_lepton_small,
        'medium': lepton_model.create_lepton_medium, 'large': lepton_model.create_lepton_large,
        'xl': lepton_model.create_lepton_xl, 'xxl': lepton_model.create_lepton_xxl,
    }
    model = creators[args.model_size]()
    print(f"LEPTON-{args.model_size}: {model.count_parameters():,} params")

    train_loader, val_loader = create_fast_dataloaders(args.data_dir, batch_size=args.batch_size, seq_len=args.seq_len)

    tokenizer_path = args.tokenizer if os.path.isabs(args.tokenizer) else os.path.join(os.path.dirname(__file__), args.tokenizer)

    trainer = LeptonTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader, tokenizer_path=tokenizer_path,
        lr=args.lr, max_steps=args.max_steps, grad_accum_steps=args.grad_accum, checkpoint_dir=args.checkpoint_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
