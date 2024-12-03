from PIL import Image
import numpy as np
import cv2
import random
import os
import tqdm
import json
import argparse
import sys
from annotator.lineart import LineartDetector
# device = 'cuda:0'

device = 'cpu'
global apply_linear

def parse_args():
    parser = argparse.ArgumentParser(description='MVEdit 3D Toolbox')
    parser.add_argument('--input_dir', default='food', type=str)
    parser.add_argument('--output_dir', default='food', type=str)
    parser.add_argument('--json', default='text_captions_cap3d.json', type=str)
    return parser.parse_args()

def to_sketch(image_path, output_path):
    image = Image.open(image_path)
    image = np.array(image)[..., :3]
    image = cv2.resize(image, (360, 240))
    # print(frame_input.shape)
    detected_map = apply_linear(image, coarse=False)
    detected_map = cv2.resize(detected_map, (512, 512))
    detected_map[detected_map < 200] = 0
    detected_map[detected_map > 200] = 255
    im = Image.fromarray(detected_map)
    im.save(output_path)
                
def main():
    succeed = 0
    failed = 0
    succeed_list = []
    failed_list = []
    args = parse_args()
    input_dir:str = args.input_dir
    output_dir:str = args.output_dir
    dirs1 = os.listdir(input_dir)
    dirs2 = []
    for dir in dirs1:
        if os.path.isdir(os.path.join(input_dir, dir)):
            path = os.path.join(input_dir, dir)
            sub_dirs = os.listdir(path)
            for sub_dir in sub_dirs:
                if os.path.isdir(os.path.join(path, sub_dir)):
                    dirs2.append(os.path.join(dir, sub_dir))
    
    for dir in dirs2:
        print('handling dir:', dir)
        try:
            output_dir_path = os.path.join(output_dir, dir)
            os.makedirs(output_dir_path, exist_ok=True)
            if input_dir != output_dir:
                for i in range(24):
                    sub_dir = f'{i:05}'
                    input_path = os.path.join(input_dir, dir, sub_dir, f'{i:05}.png')
                    frame_output_dir_path = os.path.join(output_dir_path, f'{i:05}')
                    os.makedirs(frame_output_dir_path, exist_ok=True)
                    with open(os.path.join(frame_output_dir_path, f'{i:05}.json'), 'w') as out_json:
                        with open(os.path.join(input_dir, dir, sub_dir, f'{i:05}.json'), 'r') as in_json:
                            out_json.write(in_json.read())
                    with open(os.path.join(frame_output_dir_path, f'{i:05}_nd.exr'), 'wb') as out_nd:
                        with open(os.path.join(input_dir, dir, sub_dir, f'{i:05}_nd.exr'), 'rb') as in_nd:
                            out_nd.write(in_nd.read())
                    with open(os.path.join(frame_output_dir_path, f'{i:05}.png'), 'wb') as out_png:
                        with open(os.path.join(input_dir, dir, sub_dir, f'{i:05}.png'), 'rb') as in_png:
                            out_png.write(in_nd.read())
            for i in range(1):
                sub_dir = f'{i:05}'
                input_path = os.path.join(input_dir, dir, sub_dir, f'{i:05}.png')
                frame_sk_output_dir_path = os.path.join(output_dir_path, f'{i:05}', f'sk_{i:05}.png')
                to_sketch(input_path, frame_sk_output_dir_path)
            succeed_list.append(dir)
            succeed += 1
        except Exception as e:
            print('error:', e)
            failed_list.append(dir)
            failed += 1
    
    print(f'succeed: {succeed}, failed: {failed}')
    print('failed_list:', failed_list)
    with open(f'{output_dir}/data.json', 'w') as json_file:
        json.dump(succeed_list, json_file)

if __name__ == '__main__':
    apply_linear = LineartDetector(device)
    main()