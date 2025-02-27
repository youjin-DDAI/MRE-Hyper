import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import random
import xarray as xr
from utils import load_mat_file
from Baseline.mre_pinn.data.dataset import MREExample


class MREDataModule(pl.LightningDataModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.batch_size = configs['batch_size']

    def setup(self, stage=None):
        configs = self.configs
        self.train_dataset = MREDataset(configs)
        self.valid_dataset = MREDataset_Eval(configs, self.train_dataset.data_dict)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda'))#, generator=torch.Generator(device='cuda'))

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device='cuda')) #generator=torch.Generator(device='cuda'))#, generator=torch.Generator(device='cuda'), persistent_workers=True, num_workers=8)

def make_mre_example(wave_data, mre_data, coords, i, f):
    xs, ys, zs = coords
    u = wave_data[..., i]
    u = u[..., [1, 0, 2]]  # original motion encoding direction : y, x, z
    wave_xarray = xr.DataArray(u, dims=['x', 'y', 'z', 'component'], \
                               coords={'x': xs, 'y': ys, 'z': zs, 'component': ['x', 'y', 'z']})
    mu_xarray = xr.DataArray(mre_data, dims=['x', 'y', 'z'], coords={'x': xs, 'y': ys, 'z': zs})
    example = MREExample(f, wave_xarray, mu_xarray, None)

    return example

class MREDataset(Dataset):
    def __init__(self, configs):
        dataset = configs['dataset']
        self.noise_ratio = configs['noise_ratio']
        self.batch_size = configs['batch_size']
        self.coords_shape = configs['coords_shape']
        self.num_obs = self.coords_shape[0] * self.coords_shape[1] * self.coords_shape[2]
        self.crop_x = configs['crop_size_x']
        self.crop_y = configs['crop_size_y']
        self.crop_z = configs['crop_size_z']

        # preprocessing
        self.data_dict = {}

        spacing = configs['spacing']
        freqs = configs['freqs']

        if dataset != 'fem_box':
            dis_data, _ = load_mat_file(configs['dis_data_path'])
            gt_data, _ = load_mat_file(configs['gt_data_path'])
            wave_data = dis_data[configs['dis_parsing_key']]
            mre_data = gt_data[configs['gt_parsing_key']]
            wave_data = wave_data.transpose(1, 0, 2, 3, 4)
            mre_data = mre_data.transpose(1, 0, 2)
            x_len = wave_data.shape[0]
            y_len = wave_data.shape[1]
            z_len = wave_data.shape[2]
            xs = np.linspace(0, (x_len - 1) * spacing, x_len)
            ys = np.linspace(0, (y_len - 1) * spacing, y_len)
            zs = np.linspace(0, (z_len - 1) * spacing, z_len)
            coords = (xs, ys, zs)

        for i, freq in enumerate(freqs):
            try:
                if dataset=='fem_box':
                    dataroot = './Data/BIOQIC/fem_box/'
                    example = MREExample.load_xarrays(dataroot, str(freq))
                else:
                    example = make_mre_example(wave_data, mre_data, coords, i, freq)

                if self.noise_ratio>0:
                    example.add_gaussian_noise(self.noise_ratio)

                wave = torch.tensor(example.wave.field.values(), device='cpu', dtype=torch.float32) #complex
                gt = torch.tensor(example.mre.field.values(),device='cpu', dtype=torch.float32) #complex
                pos = torch.tensor(example.wave.field.points(), dtype=torch.float32, device='cpu')

                #bone mask
                threshold = 10500
                if configs['bone_mask']:
                    bone_mask = torch.tensor(torch.abs(gt)<threshold).squeeze()
                else:
                    bone_mask = torch.ones_like(gt).bool().squeeze()
                self.data_dict[freq]={'wave_data': wave, 'gt': gt, 'pos': pos, 'example': example, 'bone_mask': bone_mask}

            except:
                print('error open {}'.format(freq))



        self.idx2freq = {}
        idx = 0
        for k, _ in self.data_dict.items():
            self.idx2freq[idx] = k
            idx+=1

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        freq = self.idx2freq[idx]
        wave = self.data_dict[freq]['wave_data'].real
        pos = self.data_dict[freq]['pos']
        gt = self.data_dict[freq]['gt'].real
        bone_mask = self.data_dict[freq]['bone_mask']

        mask = torch.zeros(self.num_obs, dtype=torch.bool, device='cpu')
        true_indices = random.sample(range(self.num_obs), self.batch_size)
        mask[true_indices] = True

        wave_crop = wave.reshape(*self.coords_shape, 3)[::self.crop_x, ::self.crop_y, ::self.crop_z, :].reshape(-1, 3)

        return pos, wave, wave_crop, gt, freq, mask, bone_mask

class MREDataset_Eval(Dataset):
    def __init__(self, configs, data_dict):
        dataset = configs['dataset']
        self.batch_size = configs['batch_size']
        self.coords_shape = configs['coords_shape']
        self.num_obs = self.coords_shape[0] * self.coords_shape[1] * self.coords_shape[2]

        self.data_dict = data_dict

        self.chunk_size = configs['chunk_size']
        self.crop_x = configs['crop_size_x']
        self.crop_y = configs['crop_size_y']
        self.crop_z = configs['crop_size_z']

        self.pos_chunks = []
        self.wave_chunks = []
        self.wave_crop_chunks = []
        self.pos_crop_chunks = []
        self.bone_mask_chunks = []
        self.freqs = []

        for k, v in self.data_dict.items():
            wave = v['wave_data'].real
            pos = v['pos']
            wave_crop = wave.reshape(*self.coords_shape, 3)[::self.crop_x, ::self.crop_y, ::self.crop_z, :].reshape(-1, 3)
            pos_crop = pos.reshape(*self.coords_shape, 3)[::self.crop_x, ::self.crop_y, ::self.crop_z, :].reshape(-1, 3)
            for i in range(self.num_obs//self.chunk_size):
                start = i * self.chunk_size
                end = (i + 1) * self.chunk_size
                pos_chunk = pos[start:end]
                wave_chunk = wave[start:end]
                bone_mask_chunk = self.data_dict[k]['bone_mask'][start:end]
                self.pos_chunks.append(pos_chunk)
                self.wave_chunks.append(wave_chunk)
                self.wave_crop_chunks.append(wave_crop)
                self.pos_crop_chunks.append(pos_crop)
                self.freqs.append(str(k))
                self.bone_mask_chunks.append(bone_mask_chunk)

            if len(pos[end:])!=0:
                start = end
                pos_chunk = pos[start:]
                wave_chunk = wave[start:]
                bone_mask_chunk = self.data_dict[k]['bone_mask'][start:]
                self.pos_chunks.append(pos_chunk)
                self.wave_chunks.append(wave_chunk)
                self.wave_crop_chunks.append(wave_crop)
                self.pos_crop_chunks.append(pos_crop)
                self.freqs.append(str(k))
                self.bone_mask_chunks.append(bone_mask_chunk)


    def __len__(self):
        return len(self.pos_chunks)

    def __getitem__(self, idx):
        wave = self.wave_chunks[idx]
        wave_crop = self.wave_crop_chunks[idx]
        pos_crop = self.pos_crop_chunks[idx]
        pos = self.pos_chunks[idx]
        freq = self.freqs[idx]
        bone_mask = self.bone_mask_chunks[idx]

        return pos, pos_crop, wave, wave_crop, freq, bone_mask






