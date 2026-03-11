import functools
import json
import os
import torch
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer

from data.dataset import JsonlDataset
from data.vocab import Vocab


def get_transforms(args, is_train: bool = True):
    """
    Train: RandomResizedCrop + HorizontalFlip + ColorJitter + RandomAffine
           → tăng tính đa dạng dữ liệu, giảm overfitting
    Val/Test: Resize + CenterCrop (deterministic)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img_size = getattr(args, 'img_size', 224)

    if is_train:
        base = [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomAutoContrast(p=0.5),   # X-ray: simulate CLAHE contrast enhancement
            transforms.RandomEqualize(p=0.3),         # histogram equalization for low-contrast lesions
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            transforms.RandomAffine(degrees=0, shear=5),
        ]
        base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(base)
    else:
        base = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
        ]
        base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(base)


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels) == list:
        for label_row in data_labels:
            if label_row == '':
                label_row = ["'Others'"]
            else:
                label_row = label_row.split(', ')

            label_freqs.update(label_row)
    else:
        pass
    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    train_path = os.path.join(args.data_path, args.Train_dset_name)

    # ---------- labels ----------
    all_labels = set()
    with open(train_path) as f:
        for line in f:
            data = json.loads(line)
            if data["label"]:
                labels = (
                    data["label"].split(', ')
                    if isinstance(data["label"], str)
                    else data["label"]
                )
                all_labels.update(labels)

    args.labels = sorted(list(all_labels))
    args.n_classes = len(args.labels)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    vocab = get_vocab(args)
    args.vocab = vocab  # needed by ImageBertEmbeddings for CLS/SEP token lookup

    # ---------- separate train/val transforms ----------
    train_transform = get_transforms(args, is_train=True)
    val_transform   = get_transforms(args, is_train=False)

    img_path = os.path.join(args.data_path, "images")

    train_dataset = JsonlDataset(
        data_path=os.path.join(args.data_path, args.Train_dset_name),
        tokenizer=tokenizer,
        transforms=train_transform,
        vocab=vocab,
        args=args,
        img_path=img_path,
        test=False,
    )

    val_dataset = JsonlDataset(
        data_path=os.path.join(args.data_path, args.Valid_dset_name),
        tokenizer=tokenizer,
        transforms=val_transform,
        vocab=vocab,
        args=args,
        img_path=img_path,
        test=True,
    )

    test_dataset = JsonlDataset(
        data_path=os.path.join(args.data_path, args.Test_dset_name),
        tokenizer=tokenizer,
        transforms=val_transform,
        vocab=vocab,
        args=args,
        img_path=img_path,
        test=True,
    )

    args.train_data_len = len(train_dataset)

    collate = functools.partial(collate_fn, args=args)

    _pw = args.n_workers > 0  # persistent_workers requires num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )

    return train_loader, val_loader, test_loader