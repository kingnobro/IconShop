import os
import time
import torch
import argparse
import numpy as np

from model.decoder import SketchDecoder
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_SAMPLE = 8
BS = 4
BBOX = 200


def sample(args, cfg):
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name'])

    sketch_decoder = SketchDecoder(
        config={
            'hidden_dim': 1024,
            'embed_dim': 512, 
            'num_layers': 16, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        pix_len=cfg['pix_len'],
        text_len=cfg['text_len'],
        num_text_token=tokenizer.vocab_size,
        word_emb_path=cfg['word_emb_path'],
        pos_emb_path=cfg['pos_emb_path'],
    )
    sketch_decoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'pytorch_model.bin')))
    sketch_decoder = sketch_decoder.to(device).eval()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    texts = [
        'img,photo'
    ]

    for text in texts:
        print(f'Generate SVG for "{text}"...')

        output_dir = os.path.join(args.output, text)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # tokenize text input
        encoded_dict = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg['text_len'],
            add_special_tokens=True,
            return_token_type_ids=False,  # for RoBERTa
        )
        tokenized_text = encoded_dict["input_ids"].squeeze()
        tokenized_text = tokenized_text.repeat(BS, 1).to(device)

        # sample SVG
        generated_svg = []
        start_time = time.time()
        while len(generated_svg) < NUM_SAMPLE:
            sample_pixels = sketch_decoder.sample(n_samples=BS, text=tokenized_text)
            generated_svg += sample_pixels
        end_time = time.time()
        print(f'Generate {len(generated_svg)} svg in {end_time - start_time} seconds')
        
        # convert token sequence into SVG
        print('Rendering...')
        gen_data = []
        for sample_pixel in generated_svg:
            gen_data += raster_svg(sample_pixel)

        print('Saving...')

        for index, data in enumerate(gen_data):
            try:
                paths = []
                for d in data:
                    path = SVGTensor.from_data(d)
                    path = SVG.from_tensor(path.data, viewbox=Bbox(BBOX))
                    path.fill_(True)
                    paths.append(path)
                path_groups = paths[0].svg_path_groups
                for i in range(1, len(paths)):
                    path_groups.extend(paths[i].svg_path_groups)
                svg = SVG(path_groups, viewbox=Bbox(BBOX))
                svg.save_svg(os.path.join(output_dir, f'{str(index).zfill(5)}.svg'))
            except Exception as err_msg:
                print(err_msg)
                continue

"""
0: SVG END
1: MASK
2: EOM
3: M
4: L
5: C
"""
def raster_svg(pixels):
    try:
        pixels -= 6  # 3 END_TOKEN + 1 SVG_END + 2 CAUSAL_TOKEN

        svg_tensors = []
        path_tensor = []
        for i, pix in enumerate(pixels):
            # COMMAND = 0
            # START_POS = [1, 3)
            # CONTROL1 = [3, 5)
            # CONTROL2 = [5, 7)
            # END_POS = [7, 9)
            if pix[0] == -3:  # Move
                cmd_tensor = np.zeros(9)
                cmd_tensor[0] = 0
                cmd_tensor[7:9] = pixels[i+2]
                start_pos = pixels[i+1]
                end_pos = pixels[i+2]
                if np.all(start_pos == end_pos) and path_tensor:
                    svg_tensors.append(torch.tensor(path_tensor))
                    path_tensor = []
                path_tensor.append(cmd_tensor.tolist())
            elif pix[0] == -2:  # Line
                cmd_tensor = np.zeros(9)
                cmd_tensor[0] = 1
                cmd_tensor[7:9] = pixels[i+1]
                path_tensor.append(cmd_tensor.tolist())
            elif pix[0] == -1:  # Curve
                cmd_tensor = np.zeros(9)
                cmd_tensor[0] = 2
                cmd_tensor[3:5] = pixels[i+1]
                cmd_tensor[5:7] = pixels[i+2]
                cmd_tensor[7:9] = pixels[i+3]
                path_tensor.append(cmd_tensor.tolist())
        svg_tensors.append(torch.tensor(path_tensor))
        return [svg_tensors]
    except Exception as error_msg:  
        print(error_msg, pixels)
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sketch_weight", type=str, required=True)
    args = parser.parse_args()

    cfg = {
        'pix_len': 512,
        'text_len': 50,

        'tokenizer_name': 'google/bert_uncased_L-12_H-512_A-8',
        'word_emb_path': 'ckpts/word_embedding_512.pt',
        'pos_emb_path': None,
    }
    
    sample(args, cfg)
