import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.ndimage


def u_compare_visualization(u_pred, u, x, index):
    if type(index) == tuple:
        x_index, y_index, z_index = index
    else:
        x_index = y_index = z_index = index

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.plot(x[:, y_index, z_index, 0], u[:, y_index, z_index, 0], label='Exact u(x)', lw=2)
    plt.plot(x[:, y_index, z_index, 0], u_pred[:, y_index, z_index, 0], '--', label='Pred u(x)', lw=2)
    plt.xlabel('x')
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x[x_index, :, z_index, 1], u[x_index, :, z_index, 1], label='Exact u(y)', lw=2)
    plt.plot(x[x_index, :, z_index, 1], u_pred[x_index, :, z_index, 1], label='Pred u(y)', lw=2)
    plt.xlabel('y')
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x[x_index, y_index, :, 2], u[x_index, y_index, :, 2], label='Exact u(z)', lw=2)
    plt.plot(x[x_index, y_index, :, 2], u_pred[x_index, y_index, :, 2], label='Pred u(z)', lw=2)
    plt.xlabel('z')
    plt.tight_layout()
    plt.legend()

    plt.show()
    return

def visualize_mu_slice(mu, fixed_axis='z', fixed_index=0, cmap='gray', log_scale=False, vmax=None, vmin=None, save_path=None):

    assert mu.ndim == 3, "input should be 3d!"
    valid_axes = {'x': 0, 'y': 1, 'z': 2}
    assert fixed_axis in valid_axes, "fixed_axis should be one of 'x', 'y', 'z'!"
    axis_idx = valid_axes[fixed_axis]

    if axis_idx == 0:
        assert 0 <= fixed_index < mu.shape[0], "fixed_index out of range of x!"
        mu_2d = mu[fixed_index, :, :]
        xlabel, ylabel = "Y-axis", "Z-axis"
    elif axis_idx == 1:
        assert 0 <= fixed_index < mu.shape[1], "fixed_index out of range of y!"
        mu_2d = mu[:, fixed_index, :]
        xlabel, ylabel = "X-axis", "Z-axis"
    else:
        assert 0 <= fixed_index < mu.shape[2], "fixed_index out of range of z!"
        mu_2d = mu[:, :, fixed_index]
        xlabel, ylabel = "X-axis", "Y-axis"

    if log_scale:
        mu_2d = np.log1p(mu_2d - np.min(mu_2d) + 1e-9)

    plt.figure(figsize=(8, 6))
    if vmax is not None:
        plt.imshow(mu_2d, cmap=cmap, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(mu_2d, cmap=cmap, aspect='auto', origin='lower')

    if save_path:
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        print(f'Image saved to {save_path}')
   
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(label="mu")
        plt.title(f"mu Visualization at {fixed_axis}={fixed_index}")
        
    plt.show()
    return 

def visualize_overlay(gt_mu, pred_mu, fixed_axis='z', fixed_index=0, cmap_gt='gray', cmap_pred='jet', alpha=0.5, blur_sigma=2, vmax=None, vmin=None, save_path=None):

    assert gt_mu.shape == pred_mu.shape
    assert gt_mu.ndim == 3
    valid_axes = {'x': 0, 'y': 1, 'z': 2}
    assert fixed_axis in valid_axes
    axis_idx = valid_axes[fixed_axis]

    if axis_idx == 0:
        assert 0 <= fixed_index < gt_mu.shape[0]
        gt_2d = gt_mu[fixed_index, :, :]
        pred_2d = pred_mu[fixed_index, :, :]
        xlabel, ylabel = "Y-axis", "Z-axis"
    elif axis_idx == 1:
        assert 0 <= fixed_index < gt_mu.shape[1]
        gt_2d = gt_mu[:, fixed_index, :]
        pred_2d = pred_mu[:, fixed_index, :]
        xlabel, ylabel = "X-axis", "Z-axis"
    else:
        assert 0 <= fixed_index < gt_mu.shape[2]
        gt_2d = gt_mu[:, :, fixed_index]
        pred_2d = pred_mu[:, :, fixed_index]
        xlabel, ylabel = "X-axis", "Y-axis"

    pred_2d_blurred = scipy.ndimage.gaussian_filter(pred_2d, sigma=blur_sigma)

    plt.figure(figsize=(8, 6))

    plt.imshow(gt_2d, cmap=cmap_gt, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

    plt.imshow(pred_2d_blurred, cmap=cmap_pred, aspect='auto', origin='lower', alpha=alpha, vmin=vmin, vmax=vmax)

    if save_path:
        plt.axis('off')
        plt.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0, transparent=True)
        print(f'Image saved to {save_path}')
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(label="mu Value")
        plt.title(f"Overlay Visualization at {fixed_axis}={fixed_index}")
    plt.show()

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

    region_1_CTE = compute_CTE_one(mu_gt, mu_pred, mask_1, mask_bg)
    region_2_CTE = compute_CTE_one(mu_gt, mu_pred, mask_2, mask_bg)
    region_3_CTE = compute_CTE_one(mu_gt, mu_pred, mask_3, mask_bg)
    region_4_CTE = compute_CTE_one(mu_gt, mu_pred, mask_4, mask_bg)

    return region_1_CTE, region_2_CTE, region_3_CTE, region_4_CTE


def compute_pearson(pred, target, bone_mask=None):
    pred = as_xarray(pred, like=target)
    if bone_mask is not None:
        bone_mask = as_xarray(bone_mask.reshape(target.shape).astype(int), like=target)
        value = xr.corr(np.abs(pred), np.abs(target), weights=bone_mask)
    else:
        value = xr.corr(np.abs(pred), np.abs(target))
    return value


def relative_l2_loss(pred, target, eps=1e-8):

    num = np.linalg.norm(pred - target, axis=1)
    denom = np.linalg.norm(target, axis=1) + eps

    loss_per_sample = num / denom
    return np.mean(loss_per_sample)


def as_xarray(a, like, suffix=None):
    '''
    Convert an array to an xarray, copying the dims and coords
    of a reference xarray.

    Args:
        a: An array to convert to xarray format.
        like: The reference xarray.
    Returns:
        An xarray with the given array values.
    '''
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if suffix is not None:
        name = like.name + suffix
    else:
        name = like.name
    return xr.DataArray(a, dims=like.dims, coords=like.coords, name=name)

def load_mat_file(mat_file, verbose=False):
    '''
    Load data set from MATLAB file.
    Args:
        mat_file: Filename, typically .mat.
        verbose: Print some info about the
            contents of the file.
    Returns:
        Loaded data in a dict-like format.
        Flag indicating MATLAB axes order.
    '''
    mat_file = str(mat_file)
    print_if(verbose, f'Loading {mat_file}')
    try:
        data = scipy.io.loadmat(mat_file)
        rev_axes = True
    except NotImplementedError as e:
        # Please use HDF reader for matlab v7.3 files
        import h5py
        data = h5py.File(mat_file)
        rev_axes = False
    except:
        print(f'Failed to load {mat_file}', file=sys.stderr)
        raise
    if verbose:
        print_mat_info(data, level=1)
    return data, rev_axes


def print_mat_info(data, level=0, tab=' '*4):
    '''
    Recursively print information
    about the contents of a data set
    stored in a dict-like format.
    '''
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(tab*level + f'{k}: {type(v)} {v.shape} {v.dtype}')
        else:
            print(tab*level + f'{k}: {type(v)}')
        if hasattr(v, 'items'):
            print_mat_info(v, level+1)
            
def print_if(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)
