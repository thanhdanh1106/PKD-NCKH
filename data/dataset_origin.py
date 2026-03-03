"""
generate dataset
"""
import os
import ast
import json
import torch
import random
import pandas as pd
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from torch.utils.data import Dataset
from transformers import BertModel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ------------------------------------------------------------------
# Augmentation - Medical Imaging Optimized (CXR)
# ------------------------------------------------------------------
def get_transforms(is_train=True):
    """
    Augmentation strategy cho chest X-ray (CXR):
      Train: RandomResizedCrop + HorizontalFlip + ColorJitter + RandomAffine
             Giúp model bất biến với góc chụp, độ sáng khác nhau giữa thiết bị
      Val/Test: Resize + CenterCrop (deterministic, không augment)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            # Scale ngẫu nhiên để học multi-scale features
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            # Flip ngang (CXR có thể flip ngang, không flip dọc)
            transforms.RandomHorizontalFlip(p=0.5),
            # Xoay nhỏ: CXR thực tế đôi khi nghiêng ±10°
            transforms.RandomRotation(degrees=10),
            # Brightness/Contrast: bù đắp khác biệt giữa máy X-ray
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            # Affine nhỏ: shear nhẹ để robustness
            transforms.RandomAffine(degrees=0, shear=5),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        
def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()

class CXRDataset(Dataset):
    def __init__(self, data_path, tokenizer, args, config, is_train=True):
        self.args = args
        self.config = config
        self.is_train = is_train
        if data_path.split('.')[-1] == 'jsonl':    
            self.data_dir = os.path.dirname(data_path)
            self.data = [json.loads(l) for l in open(data_path)]
        else:
            assert data_path.endswith(".csv")
            self.data = pd.read_csv(data_path)       
        self.data_path = data_path     
        self.seq_len = config['seq_len']
        self.tokenizer = tokenizer
        self.max_seq_len = config['max_seq_len']
        self.max_seq_len -= config['num_image_embeds']
        self.total_len = self.seq_len + self.config['num_image_embeds'] + 3
        self._tril_matrix = torch.tril(torch.ones((self.total_len, self.total_len), dtype=torch.long))
        self.vocab_stoi = self.tokenizer.vocab
        self.vocab_len = len(self.vocab_stoi)
        # Chọn augmentation dựa trên train/val/test
        self.transform = get_transforms(is_train=is_train)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_path.endswith(".jsonl"):
            origin_txt, img_path, is_aligned = self.random_pair_sampling(idx)        
        elif self.data_path.endswith(".csv"):
            image_path_list = ast.literal_eval(self.data["image"][idx])
            img_path = random.choice(image_path_list)
            text = ast.literal_eval(self.data["text"][idx])
            if isinstance(text, list):
                origin_txt = random.choice(text)
            is_aligned = 1

        # Đường dẫn tương đối cần được chuyển thành tuyệt đối
        full_img_path = os.path.join(self.data_dir, img_path) if not os.path.isabs(img_path) else img_path
        image = Image.open(full_img_path).convert("RGB")
        image = self.transform(image)  # Dùng transform đã khởi tạo sẵn (train/val)
        
        tokenized_sentence = self.tokenizer(origin_txt)
        truncate_txt(tokenized_sentence, self.seq_len)

        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                            for w in tokenized_sentence]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        if self.args.disturbing_mask:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = [-100] + txt_labels + [-100]
            txt_labels_i = [-100] * (self.config['num_image_embeds'] + 2)
        else:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = txt_labels + [-100]
            txt_labels_i = [-100] * (self.config['num_image_embeds'] + 2)

        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * (self.config['num_image_embeds'] + 2)

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]
        label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]

        input_ids.extend(padding)
        attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        
        segment = [1 for _ in range(self.seq_len + 1)]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids_tensor = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)
        is_aligned = torch.tensor(is_aligned)

        full_attn = torch.tensor((attn_masks_i + attn_masks_t),
                                 dtype=torch.long).unsqueeze(0).expand(self.total_len, self.total_len).clone()

        extended_attn_masks = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
        second_st, second_end = self.config['num_image_embeds'] + 2, self.config['num_image_embeds'] + 2 + len(input_ids)
        extended_attn_masks[:, :self.config['num_image_embeds'] + 2].fill_(1)
        extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end - second_st, :second_end - second_st])
        s2s_attn = extended_attn_masks

        mixed_lst = [full_attn, s2s_attn]

        if self.args.Mixed:
            assert (self.args.s2s_prob + self.args.bi_prob) == 1.0
            attn_masks_tensor = random.choices(mixed_lst, weights=[self.args.bi_prob, self.args.s2s_prob])[0]
        elif self.args.BAR_attn:
            extended_attn_masks[:self.config['num_image_embeds']+2, :].fill_(1)
            attn_masks_tensor = extended_attn_masks
        elif self.args.disturbing_mask:
            baseline_attn = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
            baseline_attn[:self.config['num_image_embeds'] + 2, :self.config['num_image_embeds'] + 2].fill_(1)
            baseline_attn[self.config['num_image_embeds'] + 2:, self.config['num_image_embeds'] + 2:].fill_(1)
            attn_masks_tensor = baseline_attn
        else:
            attn_masks_tensor = full_attn

        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        return cls_tok, input_ids_tensor, txt_labels, attn_masks_tensor, image, segment, is_aligned, sep_tok

    def random_word(self, tokens):
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    tokens[i] = self.vocab_stoi["[MASK]"]
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab_len)
                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(-100)

        if all(o == -100 for o in output_label):
            output_label[0] = tokens[0]
            tokens[0] = self.vocab_stoi["[MASK]"]
            
        return tokens, output_label

    def random_pair_sampling(self, idx):
        # Sửa để khớp với định dạng thực tế: {"id", "label", "text", "img"}
        data_item = self.data[idx]
        d_label = data_item['label']
        d_txt = data_item['text']  # Dùng 'text' thay vì 'caption'
        d_img = data_item['img']   # Dùng 'img' thay vì 'image'
        itm_prob = random.random()

        if itm_prob > 0.5:
            return d_txt, d_img, 1
        else:
            for _ in range(300):
                random_txt, random_label = self.get_random_line()
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_txt, d_img, 0
                else:
                    pass
            return random_txt, d_img, 0

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']  # Dùng 'text'
        label = self.data[rand_num]['label']
        return txt, label

def create_dataset(tokenizer, config, args):
    train_dataset = CXRDataset(
                data_path=config['train_dataset'],
                args=args,
                tokenizer=tokenizer,
                config=config,
                is_train=True)        # Augmentation mạnh cho train
    valid_dataset = CXRDataset(
                data_path=config['valid_dataset'],
                args=args,
                tokenizer=tokenizer,
                config=config,
                is_train=False) if config['valid_dataset'] else None   # Không augment val
    test_dataset = CXRDataset(
                data_path=config['test_dataset'],
                args=args,
                tokenizer=tokenizer,
                config=config,
                is_train=False) if config['test_dataset'] else None    # Không augment test
    return [train_dataset, valid_dataset, test_dataset]

def create_sampler(datasets, shuffles, num_gpus, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_gpus, rank=global_rank, shuffle=shuffle) if dataset else None
        samplers.append(sampler)
    return samplers     

def create_loader(datasets, samplers, batch_size, is_trains, num_workers=4):
    loaders = []
    for dataset, sampler, bs, is_train in zip(datasets, samplers, batch_size, is_trains):
        if dataset:
            if is_train:
                shuffle = (sampler is None)
                drop_last = True
            else:
                shuffle = False
                drop_last = False
            loader = DataLoader(
                dataset,
                batch_size=bs,
                num_workers=num_workers,
                pin_memory=True,          # Tăng tốc CPU→GPU transfer
                sampler=sampler,
                shuffle=shuffle,
                drop_last=drop_last,
                persistent_workers=(num_workers > 0),  # Tái dùng worker processes
                prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batch
            )
            loaders.append(loader)
        else:
            loaders.append(None)
    return loaders