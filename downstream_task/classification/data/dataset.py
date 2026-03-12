import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args, img_path=None, test=False):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.vocab = vocab
        self.args = args
        self.img_path = img_path  # Add this
        self.test = test  # Add this
        
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.n_classes = len(args.labels)
        self.text_start_token = ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms
        if not hasattr(args, 'labels'):
            # Tự động xác định labels nếu không có trong args
            all_labels = set()
            for item in self.data:
                if item["label"]:
                    labels = item["label"].split(', ') if isinstance(item["label"], str) else item["label"]
                    all_labels.update(labels)
            args.labels = sorted(list(all_labels))
            
        self.n_classes = len(args.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (
            self.text_start_token
            + self.tokenizer.tokenize(self.data[index]["text"])[
                : (self.max_seq_len - 1)
            ] + self.text_start_token
        )
        segment = torch.zeros(len(sentence))
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            if self.data[index]["label"] == '':
                self.data[index]["label"] = "'Others'"
            else:
                pass  
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"].split(', ')]
            ] = 1
        else:
            pass

        image = None
        if self.data[index]["img"]:
            img_rel = self.data[index]["img"]
            # img_rel là relative path từ repo root (e.g. data/dataset/open_i/512_3ch/1.jpg)
            # Thử trực tiếp từ CWD trước, fallback về data_dir nếu không tìm thấy
            if os.path.exists(img_rel):
                img_full_path = img_rel
            else:
                img_full_path = os.path.join(self.data_dir, img_rel)
            image = Image.open(img_full_path)
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = self.transforms(image)

        # The first SEP is part of Image Token.
        segment = segment[1:]
        sentence = sentence[1:]
        # The first segment (0) is of images.
        segment += 1

        return sentence, segment, image, label