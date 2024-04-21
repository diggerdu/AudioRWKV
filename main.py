# train the latent general source separation with pytorch lightning
# chenk.knut@bytedance.com
import os
from re import A, S
import sys
import librosa
import numpy as np
import argparse
import h5py
import math
import time
import logging
import pickle
import random
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler

from utils import create_folder, dump_config, process_idc, prepprocess_audio, init_hier_head

from sed_model import SEDWrapper, Ensemble_SEDWrapper
from models import Cnn14_DecisionLevelMax
from data_generator import SEDDataset, DESED_Dataset


from model.vit import Audio_VIT
from model.swin import Audio_SwinTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import warnings



warnings.filterwarnings("ignore")


class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle = False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = train_sampler
        )
        return train_loader
    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = eval_sampler
        )
        return eval_loader
    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = test_sampler
        )
        return test_loader
    

def save_idc():
    train_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", config.index_type + ".h5")
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    process_idc(train_index_path, config.classes_num, "full_train_idc.npy")
    process_idc(eval_index_path, config.classes_num, "eval_idc.npy")

def weight_average():
    model_ckpt = []
    model_files = os.listdir(config.esm_model_folder)
    wa_ckpt = {
        "state_dict": {}
    }

    for model_file in model_files:
        model_file = os.path.join(config.esm_model_folder, model_file)
        model_ckpt.append(torch.load(model_file, map_location="cpu")["state_dict"])
    keys = model_ckpt[0].keys()
    for key in keys:
        model_ckpt_key = torch.cat([d[key].float().unsqueeze(0) for d in model_ckpt])
        model_ckpt_key = torch.mean(model_ckpt_key, dim = 0)
        assert model_ckpt_key.shape == model_ckpt[0][key].shape, "the shape is unmatched " + model_ckpt_key.shape + " " + model_ckpt[0][key].shape
        wa_ckpt["state_dict"][key] = model_ckpt_key
    torch.save(wa_ckpt, config.wa_model_path)
def check():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    # dataset file pathes
    train_index_path = os.path.join(config.dataset_path, "hdf5s","indexes", config.index_type + ".h5")
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    train_idc = np.load(config.index_type + "_idc.npy", allow_pickle = True)
    eval_idc = np.load("eval_idc.npy", allow_pickle = True)
    exp_dir = os.path.join(config.workspace, "check_dir")
    create_folder(exp_dir)

    # import dataset SEDDataset
    dataset = SEDDataset(
        index_path=train_index_path,
        idc = train_idc,
        config = config
    )
    eval_dataset = SEDDataset(
        index_path=eval_index_path,
        idc = eval_idc,
        config = config,
        eval_mode = True
    )
    audioset_data = data_prep(dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        checkpoint_callback=False,
        deterministic=False,
        default_root_dir = exp_dir,
        gpus = device_num, 
        val_check_interval = 0.1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False
    ) 
    sed_model = Audio_VIT(
        spec_size=(config.clip_samples // config.hop_size, config.mel_bins),
        patch_size=config.patch_size,
        drop_path_rate=0.1,
        config=config
    )
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    # trainer.test(model, test_dataloaders = eval_loader)
    trainer.fit(model, audioset_data)

def esm_test():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    # dataset file pathes
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    eval_idc = np.load("eval_idc.npy", allow_pickle = True)

    # import dataset SEDDataset
    eval_dataset = SEDDataset(
        index_path=eval_index_path,
        idc = eval_idc,
        config = config,
        eval_mode = True
    )
    audioset_data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        deterministic=False,
        gpus = device_num, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    if config.model_type == "pann":
        sed_model = Cnn14_DecisionLevelMax(
        sample_rate = config.sample_rate,
        window_size = config.window_size, 
        hop_size = config.hop_size,
        mel_bins = config.mel_bins,
        fmin = config.fmin,
        fmax = config.fmax,
        classes_num = config.classes_num
    )
    elif config.model_type == "vit":
        sed_model = Audio_VIT(
            spec_size=(config.clip_samples // config.hop_size, config.mel_bins),
            patch_size=config.patch_size,
            drop_path_rate=0.1,
            config=config
        )
    elif config.model_type == "swin":
        sed_models = []
        for esm_model_path in config.esm_model_pathes:
            sed_model = Audio_SwinTransformer(
                img_size=config.swin_img_size,
                patch_size=config.swin_patch_size,
                in_chans=1,
                num_classes=config.classes_num,
                window_size=config.swin_window_size,
                config = config,
                depths = config.swin_depth,
                embed_dim = config.swin_dim,
                num_heads=config.swin_num_head
            )
            sed_wrapper = SEDWrapper(
                sed_model = sed_model, 
                config = config,
                dataset = eval_dataset
            )
            ckpt = torch.load(esm_model_path, map_location="cpu")
            sed_wrapper.load_state_dict(ckpt["state_dict"])
            sed_models.append(sed_wrapper)
    
    model = Ensemble_SEDWrapper(
        sed_models = sed_models, 
        config = config,
        dataset = eval_dataset
    )
    trainer.test(model, datamodule=audioset_data)

    


def test():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    # dataset file pathes
    if config.fl_local:
        fl_npy = np.load(config.fl_dataset, allow_pickle = True)
        # import dataset SEDDataset
        eval_dataset = DESED_Dataset(
            dataset = fl_npy,
            config = config
        )
    else:
        eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
        eval_idc = np.load("eval_idc.npy", allow_pickle = True)

        # import dataset SEDDataset
        eval_dataset = SEDDataset(
            index_path=eval_index_path,
            idc = eval_idc,
            config = config,
            eval_mode = True
        )
    audioset_data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        deterministic=False,
        gpus = device_num, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    if config.model_type == "pann":
        sed_model = Cnn14_DecisionLevelMax(
        sample_rate = config.sample_rate,
        window_size = config.window_size, 
        hop_size = config.hop_size,
        mel_bins = config.mel_bins,
        fmin = config.fmin,
        fmax = config.fmax,
        classes_num = config.classes_num
    )
    elif config.model_type == "vit":
        sed_model = Audio_VIT(
            spec_size=(config.clip_samples // config.hop_size, config.mel_bins),
            patch_size=config.patch_size,
            drop_path_rate=0.1,
            config=config
        )
    elif config.model_type == "swin":
        sed_model = Audio_SwinTransformer(
            img_size=config.swin_img_size,
            patch_size=config.swin_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.swin_window_size,
            config = config,
            depths = config.swin_depth,
            embed_dim = config.swin_dim,
            patch_stride=config.swin_stride,
            num_heads=config.swin_num_head
        )
    elif config.model_type == 'rwkv6':
        from vrwkv.backbones.vrwkv6 import VRWKV6
        sed_model = VRWKV6(
            img_size=config.swin_img_size, 
            patch_size=config.swin_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            config=config
        )     
     
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = eval_dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.test(model, datamodule=audioset_data)

    

def train():
    # set exp settings
    # device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda")
    
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    # dataset file pathes
    train_index_path = os.path.join(config.dataset_path, "hdf5s","indexes", config.index_type + ".h5")
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    train_idc = np.load(config.index_type + "_idc.npy", allow_pickle = True)
    eval_idc = np.load("eval_idc.npy", allow_pickle = True)

    # set exp folder
    c_time = "" #datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)

    # import dataset SEDDataset
    dataset = SEDDataset(
        index_path=train_index_path,
        idc = train_idc,
        # index_path=eval_index_path,
        # idc = eval_idc,
        config = config
    )
    eval_dataset = SEDDataset(
        index_path=eval_index_path,
        idc = eval_idc,
        config = config,
        eval_mode = True
    )
    audioset_data = data_prep(dataset, eval_dataset, device_num)
    checkpoint_callback = ModelCheckpoint(
        monitor = "mAP",
        filename='l-{epoch:d}-{mAP:.3f}-{mAUC:.3f}',
        save_top_k = 20,
        mode = "max"
    )
    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir = checkpoint_dir,
        gpus = device_num, 
        val_check_interval = 0.1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        resume_from_checkpoint = None, #config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0,
        #strategy=DDPPlugin(find_unused_parameters=False) if device_num > 1 else None,
    )
    if config.model_type == "pann":
        sed_model = Cnn14_DecisionLevelMax(
        sample_rate = config.sample_rate,
        window_size = config.window_size, 
        hop_size = config.hop_size,
        mel_bins = config.mel_bins,
        fmin = config.fmin,
        fmax = config.fmax,
        classes_num = config.classes_num
    )
    elif config.model_type == "vit":
        sed_model = Audio_VIT(
            spec_size=(config.clip_samples // config.hop_size, config.mel_bins),
            patch_size=config.patch_size,
            drop_path_rate=0.1,
            config=config
        )
    elif config.model_type == "swin":
        sed_model = Audio_SwinTransformer(
            img_size=config.swin_img_size,
            patch_size=config.swin_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.swin_window_size,
            config = config,
            depths = config.swin_depth,
            embed_dim = config.swin_dim,
            patch_stride=config.swin_stride,
            num_heads=config.swin_num_head
        )
    elif config.model_type == 'rwkv6':
        from vrwkv.backbones.vrwkv6 import VRWKV6
        sed_model = VRWKV6(
            img_size=config.swin_img_size, # TODO: img size
            patch_size=config.swin_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            embed_dims=192,
            num_heads=3,
            drop_path_rate=0.3,
            config=config
        )
    elif config.model_type == 'rwkv4':
        from vrwkv.backbones.vrwkv import VRWKV
        sed_model = VRWKV(
            img_size=config.img_size, # TODO: img size
            patch_size=config.patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            embed_dims=config.rwkv_embed_dim,
            num_heads=config.rwkv_num_heads,
            drop_path_rate=config.rwkv_dropout_rates,
            init_values=config.rwkv_init_values,
            post_norm=config.rwkv_post_norm,
            config=config
        ) 
        
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        head_weight, head_bias = init_hier_head(config.class_map_path, config.classes_num)
        ckpt["state_dict"]["sed_model.head.weight"] = head_weight
        ckpt["state_dict"]["sed_model.head.bias"] = head_bias
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif config.swin_pretrain_path is not None:
        ckpt = torch.load(config.swin_pretrain_path, map_location="cpu")
        # load pretrain model
        ckpt = ckpt["model"]
        found_parameters = []
        unfound_parameters = []
        model_params = dict(model.state_dict())

        for key in model_params:
            m_key = key.replace("sed_model.", "")
            if m_key in ckpt:
                # print(m_key)
                if m_key == "patch_embed.proj.weight":
                    ckpt[m_key] = torch.mean(ckpt[m_key], dim = 1, keepdim = True)
                if m_key == "head.weight" or m_key == "head.bias":
                    ckpt.pop(m_key)
                    unfound_parameters.append(key)
                    continue
                assert model_params[key].shape==ckpt[m_key].shape, "%s is not match, %s vs. %s" %(key, str(model_params[key].shape), str(ckpt[m_key].shape))
                found_parameters.append(key)
                ckpt[key] = ckpt.pop(m_key)
            else:
                unfound_parameters.append(key)
        # head_weight, head_bias = init_hier_head(config.class_map_path, config.classes_num)
        # ckpt["sed_model.head.weight"] = head_weight
        # ckpt["sed_model.head.bias"] = head_bias
        print("pretrain param num: %d \t wrapper param num: %d"%(len(found_parameters), len(ckpt.keys())))
        print("unfound parameters: ", unfound_parameters)
        model.load_state_dict(ckpt, strict = False)
        model_params = dict(model.named_parameters())
        # for k in model_params:
        #     if k in found_parameters:
        #         model_params[k].requires_grad = False
    # trainer.test(model, test_dataloaders = eval_loader)
    # trainer.test(model, datamodule=audioset_data)
    trainer.fit(model, audioset_data)



def main():
    parser = argparse.ArgumentParser(description="music auto-tagging via TS-CAM")
    subparsers = parser.add_subparsers(dest = "mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    parser_esm_test = subparsers.add_parser("esm_test")
    parser_saveidc = subparsers.add_parser("save_idc")
    parser_check = subparsers.add_parser("check")
    parser_wa = subparsers.add_parser("weight_average")
    args = parser.parse_args()
    # default settings
    logging.basicConfig(level=logging.INFO) 
    pl.utilities.seed.seed_everything(seed = config.random_seed)

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "esm_test":
        esm_test()
    elif args.mode == "save_idc":
        save_idc()
    elif args.mode == "check":
        check()
    elif args.mode == "weight_average":
        weight_average()
    else:
        raise Exception("Error Mode!")
    

if __name__ == '__main__':
    import config
    main()

