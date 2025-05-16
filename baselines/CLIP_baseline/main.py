'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import torch
import os
import pandas
import numpy as np
import tqdm
import glob
import sys
import yaml
from PIL import Image
import random
import io

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def jpeg_compress_pil(img, quality_range=(65, 100)):
    """Compress a PIL image with a random JPEG quality."""
    buffer = io.BytesIO()
    quality = random.randint(*quality_range)
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def runnig_tests(input_csv, weights_dir, models_list, device, batch_size = 1):
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size=='Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### test
    with torch.no_grad():
        
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            for k in transform_dict:
                batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index==last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()

                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            assert len(batch_id)==0
        
    return table


def runnig_tests_dw2(input_csv, weights_dir, models_list, device, batch_size=1):
    table = pandas.read_csv(input_csv)[['filename']]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = f'none_{norm_type}'
        elif patch_size == 'Clip224':
            print('input resize: Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = f'Clip224_{norm_type}'
        elif isinstance(patch_size, (tuple, list)):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = f'res{patch_size[0]}_{norm_type}'
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = f'crop{patch_size}_{norm_type}'

        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    with torch.no_grad():
        do_models = list(models_dict.keys())
        do_transforms = set(models_dict[_][0] for _ in do_models)
        print(do_models)
        print(do_transforms)
        print("Running the Tests", flush=True)

        batch_img = {k: [] for k in transform_dict}
        batch_id = []
        last_index = table.index[-1]

        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])

            # Load and degrade image safely
            try:
                img = Image.open(filename).convert('RGB')
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.BILINEAR)

                # Random crop (scale from 5/8 to full size)
                w, h = img.size
                crop_scale = random.uniform(0.625, 1.0)
                crop_w, crop_h = int(w * crop_scale), int(h * crop_scale)
                left = random.randint(0, w - crop_w) if w > crop_w else 0
                top = random.randint(0, h - crop_h) if h > crop_h else 0
                img = img.crop((left, top, left + crop_w, top + crop_h))

                # Resize to 200x200
                img = img.resize((200, 200), Image.BILINEAR)

                # JPEG compression
                img = jpeg_compress_pil(img, quality_range=(65, 100))

            except Exception as e:
                print(f"⚠️ Skipping {filename} due to error: {e}")
                continue

            for k in transform_dict:
                batch_img[k].append(transform_dict[k](img))
            batch_id.append(index)

            if len(batch_id) >= batch_size or index == last_index:
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    transform_key, model = models_dict[model_name]
                    inputs = batch_img[transform_key].clone().to(device)
                    out_tens = model(inputs).cpu().numpy()

                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        raise ValueError("Unexpected model output shape")

                    logits = np.mean(out_tens, axis=(1, 2)) if len(out_tens.shape) > 1 else out_tens

                    for ii, logit in zip(batch_id, logits):
                        table.loc[ii, model_name] = logit

                batch_img = {k: [] for k in transform_dict}
                batch_id = []

            assert len(batch_id) == 0

    return table


def runnig_tests_dw(input_csv, weights_dir, models_list, device, batch_size = 1):
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size == 'Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, (tuple, list)):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### test
    with torch.no_grad():
        
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])

            # 🆕 Early downscale added here
            img = Image.open(filename).convert('RGB')
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.BILINEAR)

            for k in transform_dict:
                batch_img[k].append(transform_dict[k](img))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index == last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    out_tens = models_dict[model_name][1](
                        batch_img[models_dict[model_name][0]].clone().to(device)
                    ).cpu().numpy()

                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            assert len(batch_id) == 0
        
    return table


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--fusion"     , '-f', type=str, help="Fusion function", default='soft_or_prob')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    
    table = runnig_tests_dw(args['in_csv'], args['weights_dir'], args['models'], args['device'])
    if args['fusion'] is not None:
        table['fusion'] = apply_fusion(table[args['models']].values, args['fusion'], axis=-1)
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file
    
