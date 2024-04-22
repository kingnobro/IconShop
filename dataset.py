import torch
import numpy as np
import random
import pandas as pd
import os
from typing import List, Tuple

from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Point

"""
0: SVG END
1: MASK
2: EOM
3: M
4: L
5: C
"""
NUM_SVG_END = 1
NUM_MASK_AND_EOM = 2
MASK = 0 + NUM_SVG_END
EOM = 1 + NUM_SVG_END
NUM_CMD_TYPES = 3
CAUSAL_PAD = 3  # 2[MASK], 1[EOM]
PIX_PAD = NUM_CMD_TYPES + NUM_MASK_AND_EOM
COORD_PAD = NUM_CMD_TYPES + NUM_MASK_AND_EOM
CMD_CURVE = 2 + NUM_MASK_AND_EOM
CMD_LINE = 1 + NUM_MASK_AND_EOM
CMD_MOVE = 0 + NUM_MASK_AND_EOM  # Finally, it will add NUM_SVG_END, so it will not overlap with EOM

# FIGR-SVG-svgo
BBOX = 200
AUG_RANGE = 3


class SketchData(torch.utils.data.Dataset):
    """ sketch dataset """
    def __init__(self, meta_file_path, svg_folder, MAX_LEN, text_len, tokenizer, require_aug):  
        self.maxlen = MAX_LEN 

        mf = pd.read_csv(meta_file_path)
        mf = mf[(1<mf.len_pix) & (mf.len_pix+NUM_SVG_END+CAUSAL_PAD<=self.maxlen)]
        self.maxlen_pix = MAX_LEN
        self.meta_file = mf
        self.svg_folder = svg_folder

        self.tokenizer = tokenizer
        self.text_len = text_len
        self.num_text_token = self.tokenizer.vocab_size

        # pixel -> xy
        pixel2xy = {}
        x=np.linspace(0, BBOX-1, BBOX)
        y=np.linspace(0, BBOX-1, BBOX)
        xx,yy=np.meshgrid(x,y)
        xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
        for pixel, xy in enumerate(xy_grid):
            pixel2xy[pixel] = xy+COORD_PAD+NUM_SVG_END
        self.pixel2xy = pixel2xy

        # causal masked model
        self.sentinel_token_expectation = 1
        self.sentinel_tokens = [MASK]
        self.eos = EOM
        self.sentinel_method = "fixed"
        self.sentinel_fixed = self.sentinel_method == "fixed"

        self.uids = sorted(list(set(mf['id'].values)))
        self.require_aug = require_aug

    def __len__(self):
        return len(self.uids)

    def prepare_batch_sketch(self, pixel_v, xy_v):
        keys = np.ones(len(pixel_v))
        padding = np.zeros(self.maxlen_pix-len(pixel_v)).astype(int)  
        pixel_v_flat = np.concatenate([pixel_v, padding], axis=0)
        pixel_v_mask = 1-np.concatenate([keys, padding]) == 1   
        padding = np.zeros((self.maxlen_pix-len(xy_v), 2)).astype(int)  
        xy_v_flat = np.concatenate([xy_v, padding], axis=0)
        return pixel_v_flat, xy_v_flat, pixel_v_mask

    def get_sentinel(self, i):
        return self.sentinel_tokens[i]

    def sentinel_masking(self, document: torch.Tensor, spans: List[Tuple[int, int]]):
        document_clone = document.clone()
        document_retrieve_mask = torch.ones_like(document_clone).to(torch.bool)

        for i, span in enumerate(spans):
            document_clone[span[0]] = self.get_sentinel(i)
            document_retrieve_mask[span[0] + 1:span[1]] = False

        return document_clone[document_retrieve_mask]
    
    def sentinel_targets(self, document: torch.Tensor, spans: List[Tuple[int, int]]):
        num_focused_tokens = sum(x[1] - x[0] for x in spans)
        num_spans = len(spans)
        target = torch.zeros(num_focused_tokens + 2 * num_spans).to(document)
        index = 0
        if self.sentinel_fixed:
            assert len(self.sentinel_tokens) == len(spans)
        else:
            assert len(self.sentinel_tokens) > len(spans)

        for i, span in enumerate(spans):
            target[index] = self.get_sentinel(i)
            index += 1
            size = span[1] - span[0]
            target[index: index + size] = document[span[0]:span[1]]
            target[index + size] = self.eos
            index = index + size + 1
        return target

    def get_spans_to_mask(self, document_length: int) -> List[Tuple[int, int]]:
        start, end = np.random.uniform(size=2)
        if end < start:
            start, end = end, start
        # round down
        start = int(start * document_length)
        # round up
        end = int(end * document_length + 0.5)
        if start == end:
            return None
        else:
            assert start < end
            return [(start, end)]

    def get_ordered_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return sorted(spans, key=lambda x: x[0])

    def __getitem__(self, index):
        uid = self.uids[index]

        rand = torch.rand(1).item()
        if rand < 0.8:
            text = self.meta_file[self.meta_file.id==uid].label.values[0] # FIGR
            text = text.split('/')
            random.shuffle(text)
            text = ','.join(text)
        elif rand < 0.85:
            text = self.meta_file[self.meta_file.id==uid].desc.values[0] # FIGR
        elif rand < 0.9:
            text = self.meta_file[self.meta_file.id==uid].desc.values[0] # FIGR
        else:
            text = ''

        encoded_dict = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_len,
            add_special_tokens=True,
            return_token_type_ids=False,  # for RoBERTa
        )
        text = encoded_dict["input_ids"].squeeze()

        svg_file = os.path.join(self.svg_folder, f'{uid}.svg')

        svg = SVG.load_svg(svg_file)

        if self.require_aug:
            dx = random.randint(-AUG_RANGE, AUG_RANGE)
            dy = random.randint(-AUG_RANGE, AUG_RANGE)
            svg.translate(Point(dx, dy))
        else:
            svg.drop_z()

        # convert to vec_data
        svg_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
        vec_data = get_vec_data(svg_tensors)

        pix_tokens = vec_data['se_pix']
        pixs = np.hstack(pix_tokens)+NUM_SVG_END
        pixs = np.concatenate((pixs, np.zeros(1).astype(int)))

        rand = torch.rand(1).item()
        if rand < 0.5:
            spans = self.get_spans_to_mask(len(pixs))
            if spans is not None:
                causal_source = self.sentinel_masking(torch.from_numpy(pixs), spans)
                causal_masked = self.sentinel_targets(torch.from_numpy(pixs), spans)
                pixs = torch.cat([causal_source, causal_masked]).numpy()

        # generate xys from pixs
        xys = []
        for pix in pixs:
            if pix < COORD_PAD + NUM_SVG_END:
                xys.append(np.array([pix, pix]))
            else:
                pix -= COORD_PAD + NUM_SVG_END
                xys.append(self.pixel2xy[pix])

        pix_seq, xy_seq, mask = self.prepare_batch_sketch(pixs, xys)
        pix_seq = torch.from_numpy(pix_seq)
        xy_seq = torch.from_numpy(xy_seq)
        return pix_seq, xy_seq, mask, text


def get_vec_data(svg_tensors):
    se_pix = []
    pix_len = 0

    for path_tensor in svg_tensors:
        path_tensor = path_tensor.round().int()
        path_tensor = torch.clip(path_tensor, min=0, max=BBOX-1)
        path_pix = []
        for i, cmd_arg_tensor in enumerate(path_tensor):
            cmd = cmd_arg_tensor[0]
            start_pos = cmd_arg_tensor[1:3].numpy()
            control1 = cmd_arg_tensor[3:5].numpy()
            control2 = cmd_arg_tensor[5:7].numpy()
            end_pos = cmd_arg_tensor[7:9].numpy()

            if cmd == 0:  # Move
                if i == 0:
                    path_pix.append(CMD_MOVE)
                    path_pix.append(num2index(end_pos) + PIX_PAD)
                    path_pix.append(num2index(end_pos) + PIX_PAD)
                else:
                    path_pix.append(CMD_MOVE)
                    path_pix.append(num2index(start_pos) + PIX_PAD)
                    path_pix.append(num2index(end_pos) + PIX_PAD)
            elif cmd == 1:  # Line
                path_pix.append(CMD_LINE)
                path_pix.append(num2index(end_pos) + PIX_PAD)
            else:  # Curve
                path_pix.append(CMD_CURVE)
                path_pix.extend([
                    num2index(control1) + PIX_PAD,
                    num2index(control2) + PIX_PAD,
                    num2index(end_pos) + PIX_PAD,
                ])
        
        pix_len += len(path_pix)
        se_pix.append(np.array(path_pix))

    num_se = len(svg_tensors)

    vec_data = {
        'len_pix': pix_len,
        'num_se': num_se,
        'se_pix': se_pix,
    }
    return vec_data


def num2index(n: np.array) -> int:
    return n[0] + n[1] * BBOX
