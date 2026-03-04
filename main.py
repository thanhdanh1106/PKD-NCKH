"""
MedViLL Pre-training - Optimized for T4 15GB VRAM
Target: ACC >= 90%, minimum Loss
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import time
import torch

from data.dataset_origin import create_dataset, create_loader
from utils import utils
from models.train_origin import MedViLL_Trainer
from transformers import AutoTokenizer


def setup_cuda():
    """Cấu hình CUDA tối ưu cho T4 GPU."""
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected, running on CPU (will be slow!)")
        return

    # T4 sử dụng FP16 Tensor Cores → bật benchmark cho CUDNN
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # T4 không hỗ trợ TF32 (chỉ Ampere+) → đảm bảo tắt để kết quả FP16 chính xác
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.empty_cache()

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} | Total VRAM: {total_vram:.1f} GB")
    print(f"CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")


def train(config, args):
    utils.set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    print("Loading dataset from:", config['train_dataset'])
    dset = create_dataset(tokenizer=tokenizer, args=args, config=config)

    print("Creating DataLoaders...")
    num_workers = int(config.get('num_workers', 4))

    pin_memory = bool(config.get('pin_memory', True))

    # Train loader
    train_data_loader = create_loader(
        [dset[0]], samplers=[None],
        batch_size=[config['batch_size']],
        is_trains=[True],
        num_workers=num_workers,
        pin_memory=pin_memory,
    )[0]

    # Validation loader (nếu có)
    val_data_loader = None
    if dset[1] is not None:
        val_data_loader = create_loader(
            [dset[1]], samplers=[None],
            batch_size=[config['batch_size']],
            is_trains=[False],
            num_workers=num_workers,
            pin_memory=pin_memory,
        )[0]
        print(f"  Train: {len(dset[0])} samples | Val: {len(dset[1])} samples")
    else:
        print(f"  Train: {len(dset[0])} samples | Val: N/A")

    print("Creating MedViLL Trainer...")
    start_time = time.time()
    trainer = MedViLL_Trainer(
        args, config,
        train_dataloader=train_data_loader,
        test_dataloader=val_data_loader
    )
    print(f"Trainer ready in {time.time() - start_time:.1f}s")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM before training: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved")

    print("\n" + "="*60)
    print("Training Start!")
    print("="*60)

    epochs = int(config['epochs'])

    for epoch in range(epochs):
        # --- Train một epoch ---
        train_loss, train_acc = trainer.train(epoch)

        # --- Validate (nếu có validation set) ---
        val_loss, val_acc = trainer.validate(epoch)

        # --- Lưu model ---
        is_best = (val_loss is not None and val_loss <= trainer.best_val_loss)
        trainer.save(epoch, args.output_path, is_best=is_best)

        status = f"Train ACC={train_acc:.2f}%"
        if val_acc is not None:
            status += f" | Val ACC={val_acc:.2f}%"
        print(f"→ Epoch {epoch:02d}: {status}\n")

        # --- Early stopping ---
        if val_loss is not None and trainer.should_early_stop(val_loss):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print("\nTraining completed!")
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM used: {peak_vram:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedViLL Pre-training (T4 Optimized)")
    parser.add_argument("--mlm_task", type=bool, default=True)
    parser.add_argument("--itm_task", type=bool, default=True)
    parser.add_argument('--BAR_attn', default=True, type=bool,
                        help="Bidirectional Auto Regressive attention mask")
    parser.add_argument('--Mixed', default=False, type=bool)
    parser.add_argument('--s2s_prob', default=1.0, type=float)
    parser.add_argument('--bi_prob', default=0.0, type=float)
    parser.add_argument('--disturbing_mask', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument("--weight_load", type=bool, default=False)
    parser.add_argument("--pre_trained_model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str,
                        default='configs/pretrain.yaml',
                        help="Path to pretrain config YAML")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Override output path (default: output/MMDD-HHmm)")
    args = parser.parse_args()

    # --- Đọc config ---
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # --- Setup output path ---
    if args.output_path is None:
        now = datetime.now().strftime('%m%d-%H%M')
        args.output_path = f'output/{now}'
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    print(f"Output path: {args.output_path}")

    # --- Setup CUDA ---
    setup_cuda()

    # --- Start training ---
    train(config, args)