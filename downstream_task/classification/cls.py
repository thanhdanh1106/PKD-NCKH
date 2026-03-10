import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *


# ------------------------------------------------------------------
# Focal Loss – giảm đóng góp của easy negatives,
# giúp model tập trung học hard examples → loss hội tụ nhanh hơn
# ------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multilabel classification.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    gamma=2 (default): bỏ qua easy examples, tập trung vào hard ones
    """
    def __init__(self, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()

def get_args(parser):
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)

    now = datetime.now()
    now = now.strftime('%Y-%m-%d')
    output_path = "output/" + str(now)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    parser.add_argument("--savedir", type=str, default=output_path)
    parser.add_argument("--save_name", type=str, default='mimic_par', help='file name to save combination of dataset and loaddir name')
    parser.add_argument("--loaddir", type=str, default='/content/drive/MyDrive/MedViLL-runimage/configs/pretrain.yaml')
    parser.add_argument("--name", type=str, default="scenario_name")

    parser.add_argument("--openi", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default='/content/drive/MyDrive/MedViLL-runimage/data/dataset/openi/Train.jsonl',
                       help="dataset path for training")
    parser.add_argument("--Train_dset_name", type=str, default='Train_253.jsonl',
                       help="train dataset for mimic")
    parser.add_argument("--Valid_dset_name", type=str, default='Test_253.jsonl',
                       help="valid dataset for mimic")

    parser.add_argument("--embed_sz", type=int, default=768, choices=[768])
    parser.add_argument("--hidden_sz", type=int, default=768, choices=[768])
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                       choices=["bert-base-uncased"])
    parser.add_argument("--init_model", type=str, default="bert-base-uncased",
                       choices=["bert-base-uncased"])

    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)  # tăng từ 0.1 → 0.2 cho regularization tốt hơn

    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)

    parser.add_argument("--freeze_img_all", type=bool, default=True)
    parser.add_argument("--freeze_txt_all", type=bool, default=True)

    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # effective batch = 16*4 = 64
    parser.add_argument("--hidden", nargs="*", type=int, default=[])

    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=bool, default=True)

    parser.add_argument("--lr", type=float, default=2e-4)    # head LR; backbone lấy 0.1x = 2e-5
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)  # giữ lại cho backward compat

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_image_embeds", type=int, default=256)

    parser.add_argument("--warmup", type=float, default=0.1)  # 10% warmup
    parser.add_argument("--weight_classes", type=int, default=1)

    # T4 GPU optimizations
    parser.add_argument("--fp16", type=bool, default=True, help="Enable AMP FP16 for T4 GPU Tensor Cores")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")

    # Added img_size for image preprocessing
    parser.add_argument("--img_size", type=int, default=224, help="size to resize images to (img_size x img_size)")
    parser.add_argument("--labels", nargs="+", default=["label1", "label2"], 
                      help="List of labels for classification")
    parser.add_argument("--label_freqs", type=dict, default={},
                      help="Label frequencies for weighted loss")

def get_criterion(args, device):
    """
    Multilabel: FocalLoss với pos_weight để xử lý class imbalance.
    Classification: CrossEntropyLoss với label_smoothing.
    """
    if args.task_type == "multilabel":
        pos_weight = None
        if args.weight_classes and hasattr(args, 'label_freqs') and args.label_freqs:
            freqs = [args.label_freqs[l] for l in args.labels]
            negative = [max(args.train_data_len - f, 1) for f in freqs]
            pos_weight = (torch.FloatTensor(negative) / torch.FloatTensor(freqs)).to(device)
        criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return criterion

def get_optimizer(model, args):
    """
    Differential learning rate:
      - Pre-trained backbone (ResNet50 + BERT encoder): lr * 0.1
      - New layers (img_embeddings projection + classifier head): lr
    Giúp fine-tune backbone nhẹ nhàng mà không làm mất pre-trained features.
    """
    enc = model.enc
    backbone_params = (
        list(enc.img_encoder.parameters())
        + list(enc.encoder.parameters())
        + list(enc.txt_embeddings.parameters())
        + list(enc.pooler.parameters())
    )
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": 0.01},
            {"params": head_params,     "lr": args.lr,       "weight_decay": 0.01},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer

def get_scheduler(optimizer, args):
    """
    Cosine Annealing với Linear Warmup:
      - Warmup 10% đầu: LR tăng tuyến tính → tránh divergence giai đoạn đầu
      - Cosine decay: giảm mượt mà đến lr_min, tốt hơn ReduceLROnPlateau
    Gọi scheduler.step() sau mỗi gradient update (không phải sau mỗi epoch).
    """
    total_steps = max(
        1,
        int(args.train_data_len / args.batch_sz / args.gradient_accumulation_steps)
        * args.max_epochs,
    )
    warmup_steps = int(total_steps * args.warmup)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

def model_eval(data, model, args, criterion, device, store_preds=False):
    with torch.no_grad():
        losses, preds, preds_bool, tgts, outAUROC = [], [], [], [], []
        for batch in data:
            loss, out, tgt = model_forward(model, args, criterion, batch, device)
            losses.append(loss.item())
            if args.task_type == "multilabel":
                pred_bool = torch.sigmoid(out).cpu().detach().numpy() > 0.5
                pred = torch.sigmoid(out).cpu().detach().numpy()
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            preds_bool.append(pred_bool)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    classACC = dict()
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        preds_bool = np.vstack(preds_bool)

        for i in range(args.n_classes):
            try:
                outAUROC.append(roc_auc_score(tgts[:, i], preds[:, i]))
            except ValueError:
                outAUROC.append(0)
                pass
        for i in range(0, len(outAUROC)):
            assert args.n_classes == len(outAUROC)
            classACC[args.labels[i]] = outAUROC[i]

        metrics["micro_roc_auc"] = roc_auc_score(tgts, preds, average="micro")
        metrics["macro_roc_auc"] = roc_auc_score(tgts, preds, average="macro")
        metrics["avg_auroc"] = float(np.mean(outAUROC))  # avg AUROC across all classes
        metrics["macro_f1"]        = f1_score(tgts, preds_bool, average="macro")
        metrics["micro_f1"]        = f1_score(tgts, preds_bool, average="micro")
        metrics["avg_f1"]          = float(np.mean([f1_score(tgts[:, i], preds_bool[:, i]) for i in range(tgts.shape[1])]))
        metrics["macro_precision"] = precision_score(tgts, preds_bool, average="macro", zero_division=0)
        metrics["micro_precision"] = precision_score(tgts, preds_bool, average="micro", zero_division=0)
        metrics["avg_precision"]   = float(np.mean([precision_score(tgts[:, i], preds_bool[:, i], zero_division=0) for i in range(tgts.shape[1])]))
        metrics["macro_recall"]    = recall_score(tgts, preds_bool, average="macro", zero_division=0)
        metrics["micro_recall"]    = recall_score(tgts, preds_bool, average="micro", zero_division=0)
        metrics["avg_recall"]      = float(np.mean([recall_score(tgts[:, i], preds_bool[:, i], zero_division=0) for i in range(tgts.shape[1])]))
        print('micro_auc:', metrics["micro_roc_auc"])
        print('avg_auroc:', metrics["avg_auroc"])
        print('micro_f1:', metrics["micro_f1"])
        print('avg_f1:', metrics["avg_f1"])
        print('avg_precision:', metrics["avg_precision"])
        print('avg_recall:', metrics["avg_recall"])
        print('-----------------------------------------------------')
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics, classACC, tgts, preds

def model_forward(model, args, criterion, batch, device):
    # Extract from dictionary (JsonlDataset output)
    txt  = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    img  = batch['image'].to(device)
    tgt  = batch['label'].to(device)

    # segment = token-type IDs (zeros for single sentence)
    segment = torch.zeros_like(txt)

    out  = model(txt, mask, segment, img)
    loss = criterion(out, tgt)
    return loss, out, tgt

def train(args):
    print("Training start!!")
    print(" # PID :", os.getpid())

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.save_name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)

    criterion = get_criterion(args, device)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    torch.save(args, os.path.join(args.savedir, "args.bin"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.loaddir, "pytorch_model.bin")):
        model.load_state_dict(torch.load(os.path.join(args.loaddir, "pytorch_model.bin")), strict=False)
        print("Loaded pre-trained model, fine-tuning.")
    else:
        print("Initializing model with random weights, training from scratch.")

    # ----------------------------------------------------------------
    # Freeze / unfreeze backbone tương ứng với args (chỉ set 1 lần)
    # freeze_img_all / freeze_txt_all = True  → trainable (requires_grad=True)
    #                                   False → frozen  (requires_grad=False)
    # ----------------------------------------------------------------
    for param in model.enc.img_encoder.parameters():
        param.requires_grad = args.freeze_img_all
    for param in model.enc.encoder.parameters():
        param.requires_grad = args.freeze_txt_all

    print("Freeze image?", args.freeze_img_all)
    print("Freeze text?", args.freeze_txt_all)
    model.to(device)
    logger.info("Training..")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # AMP GradScaler cho T4 FP16
    use_fp16 = getattr(args, 'fp16', False) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_fp16)
    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    if use_fp16:
        print("AMP FP16: ENABLED for downstream classification")

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            with autocast(enabled=use_fp16):
                loss, out, target = model_forward(model, args, criterion, batch, device)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item() * args.gradient_accumulation_steps)
            scaler.scale(loss).backward()
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # cosine warmup: step per gradient update

        model.eval()
        metrics, classACC, tgts, preds = model_eval(val_loader, model, args, criterion, device)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        csv_save_name = args.save_name
        save_path = os.path.join(args.savedir, f'{csv_save_name}.csv')
        with open(save_path, 'w', encoding='utf-8') as f:
            wr = csv.writer(f)
            key = list(classACC.keys())
            val = list(classACC.values())
            title = ['micro_auc', 'macro_auc', 'avg_auroc', 'micro_f1', 'macro_f1', 'avg_f1', 'avg_precision', 'avg_recall'] + key
            result = [metrics["micro_roc_auc"], metrics["macro_roc_auc"], metrics["avg_auroc"], metrics["micro_f1"], metrics["macro_f1"], metrics["avg_f1"], metrics["avg_precision"], metrics["avg_recall"]] + val
            wr.writerow(title)
            wr.writerow(result)

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

def test(args):
    print("Model Test")
    print(" # PID :", os.getpid())
    print('log:', args.Valid_dset_name)
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)

    criterion = get_criterion(args, device)
    torch.save(args, os.path.join(args.savedir, "args.bin"))

    if os.path.exists(os.path.join(args.loaddir, "model_best.pt")):
        model.load_state_dict(torch.load(os.path.join(args.loaddir, "model_best.pt")), strict=False)
        print("Loaded best model.")
    else:
        print("Initializing model with random weights, testing from scratch.")

    print("Freeze image?", args.freeze_img_all)
    print("Freeze text?", args.freeze_txt_all)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    load_checkpoint(model, os.path.join(args.loaddir, "model_best.pt"))

    model.eval()
    metrics, classACC, tgts, preds = model_eval(val_loader, model, args, criterion, device, store_preds=True)

    print('micro_roc_auc:', round(metrics["micro_roc_auc"], 3))
    print('macro_roc_auc:', round(metrics["macro_roc_auc"], 3))
    print('avg_auroc:', round(metrics["avg_auroc"], 3))
    print('macro_f1 f1 score:', round(metrics["macro_f1"], 3))
    print('micro f1 score:', round(metrics["micro_f1"], 3))
    print('avg_f1:', round(metrics["avg_f1"], 3))
    print('avg_precision:', round(metrics["avg_precision"], 3))
    print('avg_recall:', round(metrics["avg_recall"], 3))
    for i in classACC:
        print(i, round(classACC[i], 3))

    # --- Bootstrap p-value (so sánh model vs random baseline 0.5) ---
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    boot_auroc, boot_f1 = [], []
    n = tgts.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            b_auroc = np.mean([roc_auc_score(tgts[idx, i], preds[idx, i])
                               for i in range(tgts.shape[1])])
        except ValueError:
            b_auroc = 0.5
        b_f1 = float(np.mean([f1_score(tgts[idx, i], (preds[idx, i] > 0.5).astype(int))
                               for i in range(tgts.shape[1])]))
        boot_auroc.append(b_auroc)
        boot_f1.append(b_f1)
    # p-value: tỷ lệ bootstrap samples ≤ baseline 0.5
    p_auroc = float(np.mean(np.array(boot_auroc) <= 0.5))
    p_f1    = float(np.mean(np.array(boot_f1)    <= 0.0))
    print(f'p-value (avg AUROC): {p_auroc:.4f}')
    print(f'p-value (avg F1):    {p_f1:.4f}')

def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    
    # Sửa đường dẫn như sau:
    args.data_path = "/content/drive/MyDrive/MedViLL-runimage/data/dataset/openi"  # Chỉ đến thư mục chứa file
    args.Train_dset_name = "Train.jsonl"  # Chỉ tên file
    args.Valid_dset_name = "Test.jsonl"   # Chỉ tên file
    
    print('=========INFO==========')
    print('loaddir:', args.loaddir)
    print('openi:', args.openi)
    print('data_path:', args.data_path)
    print('train_dset:', os.path.join(args.data_path, args.Train_dset_name))  # In ra đường dẫn đầy đủ để kiểm tra
    print('valid_dset:', os.path.join(args.data_path, args.Valid_dset_name))
    print('========================')
    
    train(args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()