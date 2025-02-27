import torch
import pytorch_lightning as pl
import math
import torchvision.utils as vutils
from Model.OP import MREHyper
from Baseline.mre_pinn.pde import WaveEquation
import xarray as xr
import numpy as np
from utils import as_xarray
import matplotlib.pyplot as plt
import io
from PIL import Image
import scipy
import colorcet as cc
import pickle

def msae_loss(y_pred, y_true):
    return torch.mean(
        torch.abs((y_true - y_pred))**2
    )

def mae_loss(y_pred, y_true):
    return torch.mean(
        torch.abs((y_true - y_pred))
    )

def compute_pearson(pred, target, bone_mask=None):
    pred = pred.squeeze().cpu().detach().numpy().reshape(target.shape)
    pred = as_xarray(pred, like=target)
    if bone_mask is not None:
        bone_mask = as_xarray(bone_mask.reshape(target.shape).cpu().numpy().astype(int), like=target)
        value = xr.corr(np.abs(pred), np.abs(target.real), weights=bone_mask)
    else:
        value = xr.corr(np.abs(pred), np.abs(target.real))
    return value


class loss_pde(object):
    def __init__(self, pde_name, detach=True):
        self.pde = WaveEquation.from_name(pde_name, omega=None, detach=detach)
        self.loss_fn = msae_loss

    def __call__(self, x_loc, freq, w, mu_pred, bone_mask):
        pde_residual = self.pde(x_loc, w, mu_pred, freq)
        pde_loss = self.loss_fn(torch.zeros_like(pde_residual[bone_mask]), pde_residual[bone_mask])
        return pde_loss


def compute_CTE_one(target, pred, mask_r, mask_bg):
    target_MAC = np.mean(target[mask_r]) - np.mean(target[mask_bg])  # MAV(r)-MAV(bg)
    pred_MAC = np.mean(pred[mask_r]) - np.mean(pred[mask_bg])
    CTE = (pred_MAC / target_MAC) * 100
    return CTE


def compute_CTE(mu_pred, mu_gt, spatial_info):
    mask_bg = spatial_info == 1
    mask_1 = spatial_info == 2
    mask_2 = spatial_info == 3
    mask_3 = spatial_info == 4
    mask_4 = spatial_info == 5

    mu_pred = mu_pred.reshape(mu_gt.shape).detach().cpu().numpy()
    mu_gt = mu_gt.real.values

    region_1_CTE = compute_CTE_one(mu_gt, mu_pred, mask_1, mask_bg)
    region_2_CTE = compute_CTE_one(mu_gt, mu_pred, mask_2, mask_bg)
    region_3_CTE = compute_CTE_one(mu_gt, mu_pred, mask_3, mask_bg)
    region_4_CTE = compute_CTE_one(mu_gt, mu_pred, mask_4, mask_bg)

    return region_1_CTE, region_2_CTE, region_3_CTE, region_4_CTE

def visualize_overlay_tensorboard(writer, tag, gt_mu, pred_mu, fixed_axis='z', fixed_index=0, cmap_gt='gray', \
                                  cmap_pred='jet', alpha=0.5, blur_sigma=2, vmax=10500, vmin=0, step=0):

    pred_mu = pred_mu.reshape(*gt_mu.shape).detach().cpu().numpy()
    assert gt_mu.shape == pred_mu.shape, "gt_mu and pred_mu must have the same shape!"
    assert gt_mu.ndim == 3, "Input data must be 3-dimensional!"
    valid_axes = {'x': 0, 'y': 1, 'z': 2}
    assert fixed_axis in valid_axes, "fixed_axis must be one of 'x', 'y', or 'z'!"
    axis_idx = valid_axes[fixed_axis]

    if axis_idx == 0:
        gt_2d = gt_mu[fixed_index, :, :]
        pred_2d = pred_mu[fixed_index, :, :]
    elif axis_idx == 1:
        gt_2d = gt_mu[:, fixed_index, :]
        pred_2d = pred_mu[:, fixed_index, :]
    else:
        gt_2d = gt_mu[:, :, fixed_index]
        pred_2d = pred_mu[:, :, fixed_index]

    pred_2d_blurred = scipy.ndimage.gaussian_filter(pred_2d, sigma=blur_sigma)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(gt_2d, cmap=cmap_gt, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.imshow(pred_2d_blurred, cmap=cmap_pred, aspect='auto', origin='lower', alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_title(f"Overlay Visualization at {fixed_axis}={fixed_index}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf)
    image = np.array(image)[:, :, :3]
    writer.add_image(tag, image.transpose(2, 0, 1), step)


class LightningMRE(pl.LightningModule):
    def __init__(self, params, datamodule):
        super(LightningMRE, self).__init__()
        self.params = params
        self.bone_mask = params['bone_mask']
        self.noise_ratio = params['noise_ratio']
        self.dataset = params['dataset']
        x_full, y_full, z_full = params['coords_shape']
        x_crop = params['crop_size_x']
        y_crop = params['crop_size_y']
        z_crop = params['crop_size_z']

        reduced_num_obs = (math.floor((x_full-1)/x_crop)+1)*(math.floor((y_full-1)/y_crop)+1)*(math.floor((z_full-1)/z_crop)+1)
        self.model = MREHyper(observation_num=reduced_num_obs, freq=params['omega'])
        self.lr = float(params['lr'])
        self.weight_decay = float(params['weight_decay'])
        self.loss_pde = loss_pde(params['pde'])
        self.loss_recon = msae_loss

        self.pde_scheduling = params['pde_scheduling']
        self.pde_loss_weight = float(params['pde_loss_weight'])
        self.pde_init_weight = float(params['pde_init_weight'])
        self.pde_warmup_iters = int(params['pde_warmup_iters'])
        self.pde_step_iters = int(params['pde_step_iters'])
        self.pde_step_factor = float(params['pde_step_factor'])

        self.coords_shape = params['coords_shape']
        self.num_obs = self.coords_shape[0]*self.coords_shape[1]*self.coords_shape[2]
        self.num_freq = len(params['freqs'])
        self.chunk_size = datamodule.chunk_size

        self.data_dict = datamodule.data_dict

        self.validation_loss = []
        self.validation_pde_loss = []
        self.validation_recon_loss = []
        self.validation_u_pred = []
        self.validation_mu_pred = []
        self.validation_freqs = []

        """load_path = './tensorboard/MREHyper_fem_abdomen_1e-05/lightning_logs/version_0/epoch=19399.ckpt'

        ckpt = torch.load(load_path)
        state_dict = ckpt['state_dict']
        new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
        self.model.load_state_dict(new_state_dict)"""


    def training_step(self, batch, batch_idx):

        pos, ux, u_crop, gt_mu, freq, mask, bone_mask = batch

        pos_mask = pos[mask]
        pos_mask.requires_grad = True

        u_pred, mu_pred = self.model((u_crop, pos_mask))
        loss_pde = self.loss_pde(pos_mask, torch.full(u_pred.shape, float(freq[0])), u_pred, mu_pred, bone_mask.squeeze(0)[mask.squeeze(0)])
        loss_recon = self.loss_recon(u_pred, ux[mask])

        if self.pde_scheduling:
            pde_iter = self.global_step - self.pde_warmup_iters
            if pde_iter < 0:
                pde_weight = 0

            else:  # PDE training
                n_steps = pde_iter // self.pde_step_iters
                pde_factor = self.pde_step_factor ** n_steps
                pde_weight = min(self.pde_loss_weight, self.pde_init_weight * pde_factor)
        else:
            pde_weight = self.pde_loss_weight

        loss = pde_weight * loss_pde + loss_recon
        output_dict = {'loss': loss, 'loss_pde': loss_pde, 'loss_recon': loss_recon}
        self.log_dict(output_dict)

        return output_dict


    def validation_step(self, batch, batch_idx):

        with torch.set_grad_enabled(True):
            pos, pos_crop, ux, u_crop, freq, bone_mask = batch
            pos = pos.squeeze(0)
            pos.requires_grad = True
            u_pred, mu_pred = self.model((u_crop, pos))#self.model((u_crop, pos), float(freq[0]))#
            loss_pde = self.loss_pde(pos, torch.full(u_pred.shape, float(freq[0])), u_pred, mu_pred, bone_mask.squeeze(0))
            u_pred = u_pred.detach()
            mu_pred = mu_pred.detach()
            loss_recon = self.loss_recon(u_pred, ux)
            if self.pde_scheduling:
                pde_iter = self.global_step - self.pde_warmup_iters
                if pde_iter < 0:
                    pde_weight = 0

                else:  # PDE training
                    n_steps = pde_iter // self.pde_step_iters
                    pde_factor = self.pde_step_factor ** n_steps
                    pde_weight = min(self.pde_loss_weight, self.pde_init_weight * pde_factor)
            else:
                pde_weight = self.pde_loss_weight

        loss = pde_weight * loss_pde + loss_recon

        self.validation_loss.append(loss.item())
        self.validation_pde_loss.append(loss_pde.item())
        self.validation_recon_loss.append(loss_recon.item())
        self.validation_u_pred.append(u_pred)
        self.validation_mu_pred.append(mu_pred)
        self.validation_freqs.append(int(freq[0]))

        return

    def on_validation_epoch_end(self):
        val_loss = 0
        val_loss_pde = 0
        val_loss_recon = 0
        total_corr_mu = 0
        total_corr_w = 0

        r1_CTE = 0
        r2_CTE = 0
        r3_CTE = 0
        r4_CTE = 0

        num_chunks = math.ceil(self.num_obs/self.chunk_size)
        num_freq = len(self.validation_freqs)//num_chunks
        writer = self.logger.experiment

        any_freq = self.validation_freqs[0]
        ex = self.data_dict[any_freq]
        gt_mu = ex['example']['mre'].real.values
        if self.bone_mask:
            gt_mu = torch.from_numpy(np.minimum(gt_mu, 10500))
        else:
            gt_mu = torch.from_numpy(gt_mu)


        gt_mu_grid = vutils.make_grid(gt_mu.reshape(*self.coords_shape, 1)[:, :, 5, :], normalize=True)
        writer.add_image(f'Validation/gt_mu', gt_mu_grid.permute(2, 0, 1).cpu().numpy(),
                         self.current_epoch)
        for i in range(num_freq):
            start = i * num_chunks
            end = (i + 1) * num_chunks
            freq = self.validation_freqs[start]
            val_loss += (sum(self.validation_loss[start:end]) / num_chunks)
            val_loss_pde += (sum(self.validation_pde_loss[start:end]) / num_chunks)
            val_loss_recon += (sum(self.validation_recon_loss[start:end]) / num_chunks)
            pred_u = torch.cat(self.validation_u_pred[start:end], dim=0)
            pred_mu = torch.cat(self.validation_mu_pred[start:end], dim=0)


            ex = self.data_dict[freq]
            gt_mu = ex['example']['mre']
            gt_u = ex['example']['wave']
            if self.dataset=='fem_box':
                Region_1_CTE, Region_2_CTE, Region_3_CTE, Region_4_CTE = compute_CTE(pred_mu.squeeze(), gt_mu, gt_mu.spatial_region.values)
                r1_CTE += Region_1_CTE
                r2_CTE += Region_2_CTE
                r3_CTE += Region_3_CTE
                r4_CTE += Region_4_CTE

            if self.dataset=='fem_abdomen':
                visualize_overlay_tensorboard(writer, f'Validation/pred_mu_preq_{freq}', gt_mu.real.values, pred_mu.squeeze(), \
                                              fixed_axis='z', fixed_index=5, step=10, cmap_pred=cc.cm['rainbow_bgyr_10_90_c83'])
            else:
                img_grid = vutils.make_grid(pred_mu.reshape(*self.coords_shape, 1)[:, :, 5, :], normalize=True)
                writer.add_image(f'Validation/pred_mu_freq_{freq}', img_grid.permute(2, 0, 1).cpu().numpy(),
                                 self.current_epoch)

            corr_mu = float(compute_pearson(pred_mu.squeeze(), gt_mu, ex['bone_mask']))
            corr_u = float(compute_pearson(pred_u.squeeze(), gt_u))

            total_corr_mu += corr_mu
            total_corr_w += corr_u

            """name = 'MREHyper'
            save_file_name = './Results/'+self.dataset+'/'+name+'_preds_'+str(freq)+ '_' + str(self.noise_ratio)+'.pkl'
            with open(save_file_name, 'wb') as f:
                pickle.dump((pred_u.squeeze(),gt_u.real.values, pred_mu.squeeze(),
                             gt_mu.real.values), f,
                            pickle.HIGHEST_PROTOCOL)
            print('save_{}_pred'.format(freq))"""

        val_loss /= num_freq
        val_loss_pde /= num_freq
        val_loss_recon /= num_freq
        total_corr_mu /= num_freq
        total_corr_w /= num_freq

        r1_CTE /= num_freq
        r2_CTE /= num_freq
        r3_CTE /= num_freq
        r4_CTE /= num_freq

        self.validation_loss = []
        self.validation_pde_loss = []
        self.validation_recon_loss = []
        self.validation_u_pred = []
        self.validation_mu_pred = []
        self.validation_freqs = []

        if self.dataset == 'fem_box':
            output_dict = {'val loss': val_loss, 'val loss_pde': val_loss_pde, 'val loss_recon': val_loss_recon, \
                           'corr_mu': total_corr_mu, 'corr_w': total_corr_w, 'Mean R1 CTE': r1_CTE, 'Mean R2 CTE': r2_CTE,\
                           'Mean R3 CTE': r3_CTE, 'Mean R4 CTE': r4_CTE}
        else:
            output_dict = {'val loss': val_loss, 'val loss_pde': val_loss_pde, 'val loss_recon': val_loss_recon, \
                           'corr_mu': total_corr_mu, 'corr_w': total_corr_w}
        self.log_dict(output_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer}


