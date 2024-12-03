import os
import cv2
import json
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from tools.datasets.video_dataset import VideoDataset, read_camera_matrix_single
from utils.registry_class import DATASETS
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

@DATASETS.register_class()
class SketchVideoDataset(VideoDataset):

    def __getitem__(self, index):
        index = index % len(self.image_list)
        data_dir, file_path = self.image_list[index]
        video_key = file_path
        caption = self.captions[file_path] + ", 3d asset"

        try:
            ref_frame, vit_frame, video_data, fullreso_video_data, camera_data, mask_data, fullreso_mask_data, sketch_data = self._get_video_data(data_dir, file_path)
            if self.prepare_lgm:
                results = self.prepare_gs(camera_data.clone(), fullreso_mask_data.clone(), fullreso_video_data.clone())
                results['images_output'] = fullreso_video_data # GT renderings of [512, 512] resolution in the range [0,1]
        except Exception as e:
            print(e)
            return self.__getitem__((index+1)%len(self)) # next available data

        if self.prepare_lgm:
            return results, ref_frame, vit_frame, video_data, camera_data, mask_data, caption, video_key, sketch_data
        else:
            return ref_frame, vit_frame, video_data, camera_data, mask_data, caption, video_key, sketch_data

    def _get_video_data(self, data_dir, file_path):
        prefix = os.path.join(data_dir, file_path)

        frames_path = [os.path.join(prefix, "{:05d}/{:05d}.png".format(frame_idx, frame_idx)) for frame_idx in range(24)]
        camera_path = [os.path.join(prefix, "{:05d}/{:05d}.json".format(frame_idx, frame_idx)) for frame_idx in range(24)]
        sketch_path = os.path.join(prefix, "sk_00000.png")

        frame_list = []
        fullreso_frame_list = []
        camera_list = []
        mask_list = []
        fullreso_mask_list = []
        for frame_idx, frame_path in enumerate(frames_path):
            img = Image.open(frame_path).convert('RGBA')
            mask = torch.from_numpy(np.array(img.resize((self.resolution[1], self.resolution[0])))[:,:,-1]).unsqueeze(-1)
            mask_list.append(mask)
            fullreso_mask = torch.from_numpy(np.array(img)[:,:,-1]).unsqueeze(-1)
            fullreso_mask_list.append(fullreso_mask)

            width = img.width
            height = img.height
            # grey_scale = random.randint(128, 130) # random gray color
            grey_scale = 128 
            image = Image.new('RGB', size=(width, height), color=(grey_scale,grey_scale,grey_scale))
            image.paste(img,(0,0),mask=img)

            fullreso_frame_list.append(torch.from_numpy(np.array(image)/255.0).float()) # for LGM rendering NOTE notice the data range [0,1]
            frame_list.append(image.resize((self.resolution[1], self.resolution[0])))

            _, camera_embedding = read_camera_matrix_single(camera_path[frame_idx])
            camera_list.append(torch.from_numpy(camera_embedding.flatten().astype(np.float32)))

        camera_data = torch.stack(camera_list, dim=0) # [24,16]
        mask_data = torch.stack(mask_list, dim=0) 
        fullreso_mask_data = torch.stack(fullreso_mask_list, dim=0) 
        sketch_data = Image.open(sketch_path).convert('RGB').resize((self.resolution[1], self.resolution[0]))

        video_data = torch.zeros(self.max_frames, 3,  self.resolution[1], self.resolution[0])
        fullreso_video_data = torch.zeros(self.max_frames, 3,  512, 512)
        if self.get_first_frame:
            ref_idx = 0
        else:
            ref_idx = int(len(frame_list)/2)

        mid_frame = copy(frame_list[ref_idx])
        vit_frame = self.vit_transforms(mid_frame)
        frames = self.transforms(frame_list)
        video_data[:len(frame_list), ...] = frames
        sketch_data = self.transforms(sketch_data)

        fullreso_video_data[:len(fullreso_frame_list), ...] = torch.stack(fullreso_frame_list, dim=0).permute(0,3,1,2)

        ref_frame = copy(frames[ref_idx])
        
        return ref_frame, vit_frame, video_data, fullreso_video_data, camera_data, mask_data, fullreso_mask_data, sketch_data