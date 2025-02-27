import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from Data.UnifiedMREDataModule import MREDataModule
from UnifiedLightningModule import LightningMRE
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml
import math
def main(params):
    seed_everything(42, workers=True)
    config_file = './Configs/'+params.dataset+'.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)

    config['bone_mask'] = params.bone_mask
    config['noise_ratio'] = params.noise_ratio

    logger = TensorBoardLogger(save_dir=config['log_dir'] + '_' + str(params.noise_ratio))

    data = MREDataModule(config)
    data.setup()

    model = LightningMRE(config, data.valid_dataset)

    x, y, z = config['coords_shape']
    num_sanity_val_steps = math.ceil((x*y*z)/config['chunk_size'])
    if params.result:
        num_sanity_val_steps*=len(config['freqs'])

    if params.debug:
        trainer = pl.Trainer(accelerator="gpu", logger=None, devices=[0], max_epochs=config['max_epoch'], \
                             num_sanity_val_steps=num_sanity_val_steps, enable_checkpointing=False) # 80, 145, 95
        trainer.fit(model, datamodule=data)

    else:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=False,
            monitor="val loss_pde",
            mode="min",
            dirpath=logger.log_dir,
            filename="{epoch}",
        )

        trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=logger, max_epochs=config['max_epoch'], \
                             num_sanity_val_steps=num_sanity_val_steps, check_val_every_n_epoch=200, callbacks=[checkpoint_callback]) # 80, 145, 95
        trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fem_box')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--result', type=bool, default=False)
    parser.add_argument('--bone-mask', action='store_true', help="Enable bone mask")
    parser.add_argument('--noise-ratio', type=float, default=0.0)
    args = parser.parse_args()
    main(args)