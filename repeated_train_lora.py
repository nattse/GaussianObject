import argparse
import os
import torch
import pytorch_lightning as pl

from functools import partial
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger, LoraCheckpoint
from dataset_lora import GSCacheDataset
from torch.utils.data import DataLoader
from minlora import add_lora, LoRAParametrization
import re
_ = torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process experiment parameters.')

    parser.add_argument('--model_name', type=str, default='control_v11f1e_sd15_tile')
    parser.add_argument('--sh_degree', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--resolution', type=int, default=4)
    parser.add_argument('--sparse_num', type=int, default=8)
    parser.add_argument('--trainlist', type=str, default='test-res4-run0_fullrun test-res4-run1_fullrun')
    parser.add_argument('--exp_name', type=str, default=f'controlnet_finetune/repeated_train_lora')
    #parser.add_argument('--gs_dir', type=str, default=f'output/gs_init/kitchen')
    #parser.add_argument('--data_dir', type=str, default=f'data/mip360/kitchen')
    #parser.add_argument('--loo_dir', type=str, default=f'output/gs_init/kitchen_loo')
    parser.add_argument('--prompt', type=str, default='xxy5syt00')
    parser.add_argument('--bg_white', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--sd_locked', action='store_true', default=False)
    parser.add_argument('--only_mid_control', action='store_true', default=False)
    parser.add_argument('--train_lora', action='store_true', default=False)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--callbacks_every_n_train_steps', type=int, default=600)
    parser.add_argument('--max_steps', type=int, default=1800)
    parser.add_argument('--use_prompt_list', action='store_true', default=False)
    parser.add_argument('--manual_noise_reduce_start', type=int, default=100)
    parser.add_argument('--manual_noise_reduce_gamma', type=float, default=0.995)
    parser.add_argument('--cache_max_iter', type=int, default=100)
    parser.add_argument('--add_diffusion_lora', action='store_true', default=False)
    parser.add_argument('--add_control_lora', action='store_true', default=False)
    parser.add_argument('--add_clip_lora', action='store_true', default=False)
    parser.add_argument('--use_dust3r', action='store_true', default=False)
    parser.add_argument('--natedebug', action='store_true', help='activate vscode debugger')

    args = parser.parse_args()

    if args.natedebug:
        import debugpy
        debugpy.listen(5678)
        debugpy.wait_for_client() 
    model = None
    exp_path = os.path.join('./output', args.exp_name)

    #trainList = args.trainlist.split(' ') if not ',' in args.trainlist else args.trainlist.split(',')
    trainList = [i for i in os.listdir('/home/leelab/Desktop/GaussianObject/output/gs_init') if i.startswith('SR') and not i.endswith('_loo')]
    for tData in trainList:
        if model is None:
            model = create_model(f'./models/{args.model_name}.yaml').cpu()
            model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cpu'), strict=False)
            model.load_state_dict(load_state_dict(f'./models/{args.model_name}.pth', location='cpu'), strict=False)
        else:
            model = create_model(f'./models/{args.model_name}.yaml').cpu()
            model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cpu'), strict=False)
            model.load_state_dict(load_state_dict(f'./models/{args.model_name}.pth', location='cpu'), strict=False)
            last_models = os.listdir(f'{exp_path}/ckpts-lora')
            last_models = sorted(last_models, key = lambda x: int(re.search('lora-step=(\d+).ckpt', x)[1]), reverse = True)
            model.load_state_dict(load_state_dict(f'{exp_path}/ckpts-lora/{last_models[0]}'), strict = False)

        model.learning_rate = args.learning_rate
        model.sd_locked = args.sd_locked
        model.only_mid_control = args.only_mid_control
        model.train_lora = args.train_lora

        lora_config = {
            nn.Embedding: {
                "weight": partial(LoRAParametrization.from_embedding, rank=args.lora_rank)
            },
            nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=args.lora_rank)
            },
            nn.Conv2d: {
                "weight": partial(LoRAParametrization.from_conv2d, rank=args.lora_rank)
            }
        }

        if args.add_diffusion_lora:
            for name, module in model.model.diffusion_model.named_modules():
                if name.endswith('transformer_blocks'):
                    add_lora(module, lora_config=lora_config)
        if args.add_control_lora:
            for name, module in model.control_model.named_modules():
                if name.endswith('transformer_blocks'):
                    add_lora(module, lora_config=lora_config)
        if args.add_clip_lora:
            add_lora(model.cond_stage_model, lora_config=lora_config)

        gs_dir = os.path.join('output', 'gs_init', tData)
        data_dir = os.path.join('data', 'natedata', tData)
        loo_dir = os.path.join('output', 'gs_init', f'{tData}_loo')
        dataset = GSCacheDataset(
            gs_dir, data_dir, loo_dir,
            prompt=args.prompt,
            bg_white=args.bg_white,
            train=True,
            manual_noise_reduce_gamma=args.manual_noise_reduce_gamma,
            manual_noise_reduce_start=args.manual_noise_reduce_start,
            sh_degree=args.sh_degree,
            image_size=args.image_size,
            resolution=args.resolution,
            sparse_num=args.sparse_num,
            use_prompt_list=args.use_prompt_list,
            cache_max_iter=args.cache_max_iter,
            use_dust3r=args.use_dust3r,
        )
        dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
        loggers = [
            TensorBoardLogger(os.path.join(exp_path, 'tf_logs'))
        ]
        callbacks = [
            ImageLogger(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps, \
                        log_images_kwargs = {"plot_diffusion_rows": True, "sample": True}),
            LoraCheckpoint(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps)
        ]
        #breakpoint()
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            precision=32,
            logger=loggers,
            callbacks=callbacks,
            max_steps=args.max_steps,
            check_val_every_n_epoch=args.callbacks_every_n_train_steps//len(dataset)*2
        )

        trainer.fit(model, dataloader)
