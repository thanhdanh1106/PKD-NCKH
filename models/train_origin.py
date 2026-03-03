import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
MedViLL Trainer - Optimized for T4 15GB VRAM
Improvements:
  - AMP FP16 (torch.cuda.amp) → ~2x speedup, ~50% VRAM reduction
  - Combined MLM + ITM loss (cả 2 task thay vì bỏ MLM)
  - Cosine Annealing LR Scheduler với Linear Warmup
  - Gradient Clipping (max_norm=1.0)
  - Label Smoothing (CrossEntropyLoss với smoothing=0.1)
  - Validation loop + Early Stopping
  - VRAM usage monitoring
  - Fixed: MedViLL không còn tạo encoder kép
"""
import tqdm
import math
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig, AutoConfig
from models.MedViLL_origin import MedViLL


# ------------------------------------------------------------------
# Loss Functions: sử dụng PyTorch built-in (nhanh hơn, ổn định hơn)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Cosine Annealing với Linear Warmup
# ------------------------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """
    Linear warmup → Cosine decay đến min_lr_ratio * base_lr.
    Chuẩn mực hiện đại cho pre-training Transformer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# MedViLL Trainer
# ------------------------------------------------------------------
class MedViLL_Trainer:
    def __init__(self, args, configs, train_dataloader, test_dataloader=None):
        print("Initializing MedViLL_Trainer (T4-optimized)...")
        self.args = args
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- FP16 AMP: chỉ bật trên GPU ---
        self.use_fp16 = configs.get('fp16', False) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_fp16)
        print(f"Device: {self.device} | AMP FP16: {self.use_fp16}")

        # --- Tạo model ---
        if args.weight_load and args.pre_trained_model_path:
            print(f"Loading pre-trained model from {args.pre_trained_model_path} ...")
            model_config = AutoConfig.from_pretrained(args.pre_trained_model_path, attn_implementation="eager")
            model_state_dict = torch.load(
                os.path.join(args.pre_trained_model_path, 'pytorch_model.bin'),
                map_location='cpu'
            )
            self.model = MedViLL.from_pretrained(
                args.pre_trained_model_path,
                state_dict=model_state_dict,
                model_config=model_config,
                args=args,
                configs=configs
            )
        else:
            print("Creating new MedViLL from bert-base-uncased ...")
            model_config = BertConfig.from_pretrained("bert-base-uncased", attn_implementation="eager")
            self.model = MedViLL(model_config, args, configs)

        # Bật gradient checkpointing nếu config cho phép (giảm ~40% VRAM)
        if configs.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing: ENABLED (saves ~40% VRAM)")

        self.model = self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

        # --- Dữ liệu ---
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # --- Loss functions (built-in, nhanh hơn custom impl) ---
        smoothing = configs.get('label_smoothing', 0.1)
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=smoothing)
        self.itm_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.mlm_weight = configs.get('mlm_weight', 1.0)
        self.itm_weight = configs.get('itm_weight', 1.0)

        # --- Optimizer với weight decay chỉ áp dụng cho parameters cần thiết ---
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": float(configs.get('weight_decay', 0.01)),
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(configs['lr']),
            betas=(float(configs.get('beta1', 0.9)), float(configs.get('beta2', 0.999))),
            eps=float(configs.get('eps', 1e-6)),
        )

        # --- Cosine LR Scheduler với Linear Warmup ---
        grad_accum = int(configs.get('gradient_accumulation_steps', 1))
        num_epochs = int(configs.get('epochs', 10))
        steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
        total_steps = steps_per_epoch * num_epochs

        if configs.get('warmup_steps', 0) > 0:
            warmup_steps = int(configs['warmup_steps'])
        else:
            warmup_steps = int(total_steps * float(configs.get('warmup', 0.1)))

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=float(configs.get('min_lr', 1e-6)) / float(configs['lr'])
        )
        print(f"Total training steps: {total_steps} | Warmup steps: {warmup_steps}")

        self.max_grad_norm = float(configs.get('max_grad_norm', 1.0))
        self.grad_accum_steps = grad_accum
        self.step_cnt = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    # ------------------------------------------------------------------
    def _log_vram(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  VRAM: {alloc:.2f}GB allocated / {reserved:.2f}GB reserved")

    # ------------------------------------------------------------------
    def train(self, epoch):
        self.model.train()
        train_losses, mlm_losses, itm_losses = [], [], []
        total_correct, total_element = 0, 0

        train_data_iter = tqdm.tqdm(
            enumerate(self.train_data),
            desc=f'Epoch {epoch:02d} [Train]',
            total=len(self.train_data),
            bar_format='{l_bar}{r_bar}'
        )

        self.optimizer.zero_grad()

        for i, data in train_data_iter:
            images     = data[4].to(self.device, non_blocking=True)
            labels     = data[6].to(self.device, non_blocking=True)
            cls_tok    = data[0].to(self.device, non_blocking=True)
            input_txt  = data[1].to(self.device, non_blocking=True)
            attn_mask  = data[3].to(self.device, non_blocking=True)
            segment    = data[5].to(self.device, non_blocking=True)
            sep_tok    = data[7].to(self.device, non_blocking=True)
            txt_labels = data[2].to(self.device, non_blocking=True)  # MLM labels

            # ---- Forward với AMP autocast ----
            with autocast('cuda', enabled=self.use_fp16):
                mlm_output, itm_output = self.model(cls_tok, input_txt, attn_mask, segment, images, sep_tok)

                # ITM loss: binary classification (aligned vs not-aligned)
                itm_loss = self.itm_criterion(itm_output, labels)

                # MLM loss: token prediction (chỉ masked tokens, ignore_index=-100)
                # mlm_output: [B, seq_len, vocab_size] — reshape thành [B*seq_len, vocab_size]
                mlm_output_flat = mlm_output.view(-1, mlm_output.size(-1))
                mlm_labels_flat = txt_labels[:, self.configs['num_image_embeds'] + 2:].contiguous().view(-1)
                mlm_loss = self.mlm_criterion(mlm_output_flat, mlm_labels_flat)

                # Combined loss
                loss = self.itm_weight * itm_loss + self.mlm_weight * mlm_loss
                loss = loss / self.grad_accum_steps  # Scale cho gradient accumulation

            # ---- Backward với AMP scaler ----
            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_data):
                # Gradient clipping (unscale trước rồi clip)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()  # scheduler SAU optimizer.step()
                self.step_cnt += 1

            actual_loss = loss.item() * self.grad_accum_steps
            train_losses.append(actual_loss)
            mlm_losses.append(mlm_loss.item())
            itm_losses.append(itm_loss.item())

            correct = itm_output.argmax(dim=-1).eq(labels).sum().item()
            total_correct += correct
            total_element += labels.numel()

            # Cập nhật progress bar
            train_data_iter.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'itm': f'{itm_loss.item():.4f}',
                'mlm': f'{mlm_loss.item():.4f}',
                'acc': f'{total_correct/total_element*100:.2f}%',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })

        avg_loss = np.mean(train_losses)
        avg_acc = round(total_correct / total_element * 100, 3)
        print(f"\n[Epoch {epoch}] Train Loss: {avg_loss:.4f} | "
              f"ITM Loss: {np.mean(itm_losses):.4f} | MLM Loss: {np.mean(mlm_losses):.4f} | "
              f"Accuracy: {avg_acc}%")
        self._log_vram()
        return avg_loss, avg_acc

    # ------------------------------------------------------------------
    def validate(self, epoch):
        if self.test_data is None:
            return None, None

        self.model.eval()
        val_losses, mlm_losses, itm_losses = [], [], []
        total_correct, total_element = 0, 0

        with torch.no_grad():
            val_iter = tqdm.tqdm(
                enumerate(self.test_data),
                desc=f'Epoch {epoch:02d} [Val]',
                total=len(self.test_data),
                bar_format='{l_bar}{r_bar}'
            )
            for i, data in val_iter:
                images     = data[4].to(self.device, non_blocking=True)
                labels     = data[6].to(self.device, non_blocking=True)
                cls_tok    = data[0].to(self.device, non_blocking=True)
                input_txt  = data[1].to(self.device, non_blocking=True)
                attn_mask  = data[3].to(self.device, non_blocking=True)
                segment    = data[5].to(self.device, non_blocking=True)
                sep_tok    = data[7].to(self.device, non_blocking=True)
                txt_labels = data[2].to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_fp16):
                    mlm_output, itm_output = self.model(cls_tok, input_txt, attn_mask, segment, images, sep_tok)
                    itm_loss = self.itm_criterion(itm_output, labels)
                    mlm_output_flat = mlm_output.view(-1, mlm_output.size(-1))
                    mlm_labels_flat = txt_labels[:, self.configs['num_image_embeds'] + 2:].contiguous().view(-1)
                    mlm_loss = self.mlm_criterion(mlm_output_flat, mlm_labels_flat)
                    loss = self.itm_weight * itm_loss + self.mlm_weight * mlm_loss

                val_losses.append(loss.item())
                mlm_losses.append(mlm_loss.item())
                itm_losses.append(itm_loss.item())

                correct = itm_output.argmax(dim=-1).eq(labels).sum().item()
                total_correct += correct
                total_element += labels.numel()

        avg_loss = np.mean(val_losses)
        avg_acc = round(total_correct / total_element * 100, 3)
        print(f"[Epoch {epoch}] Val   Loss: {avg_loss:.4f} | "
              f"ITM Loss: {np.mean(itm_losses):.4f} | MLM Loss: {np.mean(mlm_losses):.4f} | "
              f"Accuracy: {avg_acc}%")
        return avg_loss, avg_acc

    # ------------------------------------------------------------------
    def should_early_stop(self, val_loss):
        """Returns True nếu nên dừng sớm."""
        patience = int(self.configs.get('patience', 5))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            print(f"No improvement ({self.patience_counter}/{patience}). Best val loss: {self.best_val_loss:.4f}")
            return self.patience_counter >= patience

    # ------------------------------------------------------------------
    def save(self, epoch, file_path, is_best=False):
        save_path_per_ep = os.path.join(file_path, str(epoch))
        os.makedirs(save_path_per_ep, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        # Lưu bằng torch.save (tránh lỗi tied weights với save_pretrained)
        model_path = os.path.join(save_path_per_ep, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), model_path)

        # Lưu config để có thể load lại
        if hasattr(model_to_save, 'config'):
            model_to_save.config.save_pretrained(save_path_per_ep)

        # Lưu optimizer/scheduler state để resume training
        torch.save({
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, os.path.join(save_path_per_ep, 'training_state.pth'))

        tag = " [BEST]" if is_best else ""
        print(f"Saved epoch {epoch} → {save_path_per_ep}{tag}")

        if is_best:
            best_path = os.path.join(file_path, 'best_model')
            os.makedirs(best_path, exist_ok=True)
            torch.save(model_to_save.state_dict(), os.path.join(best_path, 'pytorch_model.bin'))
            if hasattr(model_to_save, 'config'):
                model_to_save.config.save_pretrained(best_path)
            print(f"Best model saved → {best_path}")
