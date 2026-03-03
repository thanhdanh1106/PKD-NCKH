import os
import csv
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *

def get_args(parser):
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_sz", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=10)
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
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)

    parser.add_argument("--freeze_img_all", type=bool, default=True)
    parser.add_argument("--freeze_txt_all", type=bool, default=True)

    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])

    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=bool, default=True)

    parser.add_argument("--lr", type=float, default=5e-5)    # Tăng từ 1e-4 → 5e-5 với cosine scheduler
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)  # Giảm patience để scheduler phản ứng nhanh hơn

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_image_embeds", type=int, default=256)

    parser.add_argument("--warmup", type=float, default=0.1)
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
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            negative = [args.train_data_len - l for l in freqs]
            label_weights = (torch.FloatTensor(freqs) / torch.FloatTensor(negative)) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer(model, args):
    total_steps = (
        args.train_data_len
        / args.batch_sz
        / args.gradient_accumulation_steps
        * args.max_epochs
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer

def get_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
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
        metrics["macro_f1"] = f1_score(tgts, preds_bool, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds_bool, average="micro")
        print('micro_auc:', metrics["micro_roc_auc"])
        print('micro_f1:', metrics["micro_f1"])
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
    txt = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    img = batch['image'].to(device)
    tgt = batch['label'].to(device)
    
    # Create segment (token type IDs, usually zeros for single sentences)
    segment = torch.zeros_like(txt).to(device)

    # Handle model freezing
    model.to(device)
    if args.num_image_embeds > 0:
        for param in (model.module.enc.img_encoder.parameters() if hasattr(model, 'module') else model.enc.img_encoder.parameters()):
            param.requires_grad = args.freeze_img_all
    for param in (model.module.enc.encoder.parameters() if hasattr(model, 'module') else model.enc.encoder.parameters()):
        param.requires_grad = args.freeze_txt_all

    # Forward pass (positional arguments for MultimodalBertClf)
    out = model(txt, mask, segment, img)

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

        model.eval()
        metrics, classACC, tgts, preds = model_eval(val_loader, model, args, criterion, device)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
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
            title = ['micro_auc', 'macro_auc', 'micro_f1', 'macro_f1'] + key
            result = [metrics["micro_roc_auc"], metrics["macro_roc_auc"], metrics["micro_f1"], metrics["macro_f1"]] + val
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
    print('macro_f1 f1 score:', round(metrics["macro_f1"], 3))
    print('micro f1 score:', round(metrics["micro_f1"], 3))
    for i in classACC:
        print(i, round(classACC[i], 3))

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