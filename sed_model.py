import numpy as np
import librosa
import os
import sys
import math
import bisect
import pickle
from numpy.lib.function_base import average
from sklearn import metrics
import soundfile as sf
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

# from .utils import get_loss_func, get_mix_lambda, d_prime
from utils import get_loss_func, get_mix_lambda, d_prime
import tensorboard
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.distributed as dist
from torchlibrosa.stft import STFT, ISTFT, magphase
import pytorch_lightning as pl
# from .utils import do_mixup, get_mix_lambda, do_mixup_label
from utils import do_mixup, get_mix_lambda, do_mixup_label
import random

from torchcontrib.optim import SWA

class SEDWrapper(pl.LightningModule):
    def __init__(self, sed_model, config, dataset):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.dataset = dataset
        self.loss_func = get_loss_func(config.loss_type)

    def evaluate_metric(self, pred, ans):
        ap = []
        mAP = np.mean(average_precision_score(ans, pred, average = None))
        mAUC = np.mean(roc_auc_score(ans, pred, average = None))
        dprime = d_prime(mAUC)
        return {"mAP": mAP, "mAUC": mAUC, "dprime": dprime}
        
    def forward(self, x, mix_lambda = None):
        output_dict = self.sed_model(x, mix_lambda)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.device_type = next(self.parameters()).device
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.sed_model(x, None, True)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        # mix_lambda = None
        # if random.random() < 0.5:
        mix_lambda = torch.from_numpy(get_mix_lambda(0.5, len(batch["waveform"]))).to(self.device_type)
            # TODO: there should be a mixup in the target
            # batch["target"] = do_mixup_label(batch["target"])
            # batch["target"] = do_mixup(batch["target"], mix_lambda)
        pred, _ = self(batch["waveform"], mix_lambda)
        loss = self.loss_func(pred, batch["target"])
        # __import__('remote_pdb').set_trace()
        self.log("loss", loss, on_epoch= True, prog_bar=True)
        return loss
    def training_epoch_end(self, outputs):
        # Change: SWA
        # for opt in self.trainer.optimizers:
        #     if not type(opt) is SWA:
        #         continue
        #     opt.swap_swa_sgd()
        
        self.dataset.generate_queue()


    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
        return [pred.detach(), batch["target"].detach()]
        # return [pred.detach().cpu().numpy(), batch["target"].detach().cpu().numpy()]
    
    def validation_epoch_end(self, validation_step_outputs):
        self.device_type = next(self.parameters()).device
        # pred = np.concatenate([d[0] for d in validation_step_outputs], axis = 0)
        # target = np.concatenate([d[1] for d in validation_step_outputs], axis = 0)
        pred = torch.cat([d[0] for d in validation_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in validation_step_outputs], dim = 0)
        gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
        dist.barrier()
        metric_dict = {
            "mAP": 0.,
            "mAUC": 0.,
            "dprime": 0.
        }
        dist.all_gather(gather_pred, pred)
        dist.all_gather(gather_target, target)
        if dist.get_rank() == 0:
            gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
            gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
            metric_dict = self.evaluate_metric(gather_pred, gather_target)
            print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
            if os.path.isfile("metric.pkl"):
                os.remove("metric.pkl")
            with open("metric.pkl", "wb") as f:
                pickle.dump(metric_dict, f)
                
        else:
            metric_dict = [None]

        dist.barrier()
        with open("metric.pkl", "rb") as f:
            metric_dict = pickle.load(f)
        dist.barrier()

        self.log("mAP", metric_dict["mAP"], on_epoch = True, prog_bar=True, sync_dist=False)
        self.log("mAUC", metric_dict["mAUC"], on_epoch = True, prog_bar=True, sync_dist=False)
        self.log("dprime", metric_dict["dprime"], on_epoch = True, prog_bar=True, sync_dist=False)
        dist.barrier()
        
    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        # shift_len = random.randint(0, self.config.shift_max - 1)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        # batch["waveform"] = self.time_add(batch["waveform"], k = 3)
        # pred, pred_map = self(batch["waveform"])
        preds = []
        for i in range(20):
            pred, pred_map = self(batch["waveform"])
            preds.append(pred.unsqueeze(0))
            batch["waveform"] = self.time_shifting(batch["waveform"], shift_len = 100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred = preds.mean(dim = 0)
        if self.config.fl_local:
            return [
                pred.detach().cpu().numpy(), 
                pred_map.detach().cpu().numpy(),
                batch["audio_name"],
                batch["real_len"].cpu().numpy()
            ]
        else:
            return [pred.detach(), batch["target"].detach()]
            
            # return [
            #     pred.detach().cpu().numpy(), 
            #     batch["target"].detach().cpu().numpy(), 
            #     pred_map.detach().cpu().numpy(),
            #     batch["audio_name"],
            #     batch["hdf5_path"],
            #     batch["index_in_hdf5"].detach().cpu().numpy()
            # ]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device
        if self.config.fl_local:
            pred = np.concatenate([d[0] for d in test_step_outputs], axis = 0)
            pred_map = np.concatenate([d[1] for d in test_step_outputs], axis = 0)
            audio_name = np.concatenate([d[2] for d in test_step_outputs], axis = 0)
            real_len = np.concatenate([d[3] for d in test_step_outputs], axis = 0)
            heatmap_file = os.path.join(self.config.heatmap_dir, self.config.test_file + "_" + str(self.device_type) + ".npy")
            # print(pred.shape)
            # print(pred_map.shape)
            # print(real_len.shape)
            save_npy = [
                {
                    "audio_name": audio_name[i],
                    "heatmap": pred_map[i],
                    "pred": pred[i],
                    "real_len":real_len[i]
                }
                for i in range(len(pred))
            ]
            np.save(heatmap_file, save_npy)
        else:

            self.device_type = next(self.parameters()).device
            # pred = np.concatenate([d[0] for d in validation_step_outputs], axis = 0)
            # target = np.concatenate([d[1] for d in validation_step_outputs], axis = 0)
            pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
            target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
            gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
            dist.barrier()
            metric_dict = {
                "mAP": 0.,
                "mAUC": 0.,
                "dprime": 0.
            }
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
                # save the npy
                # heatmap_file = os.path.join(self.config.heatmap_dir, self.config.test_file + "_" + str(self.device_type) + ".npy")
                # save_npy = [
                #     {
                #         # "audio_name": audio_name[i],
                #         # "hdf5_path": hdf5_path[i],
                #         # "index_in_hdf5": index_in_hdf5[i],
                #         # "heatmap": pred_map[i],
                #         "target": gather_target[i],
                #         "pred":  gather_pred[i]
                #     }
                #     for i in range(len(gather_pred))
                # ]
                # np.save(heatmap_file, save_npy)
            self.log("mAP", metric_dict["mAP"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            self.log("mAUC", metric_dict["mAUC"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            self.log("dprime", metric_dict["dprime"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            dist.barrier()

        if False:
            pred = np.concatenate([d[0] for d in test_step_outputs], axis = 0)
            target = np.concatenate([d[1] for d in test_step_outputs], axis = 0)
            # pred_map = np.concatenate([d[2] for d in test_step_outputs], axis = 0)
            # audio_name = np.concatenate([d[3] for d in test_step_outputs], axis = 0)
            # hdf5_path = np.concatenate([d[4] for d in test_step_outputs], axis = 0)
            # index_in_hdf5 = np.concatenate([d[5] for d in test_step_outputs], axis = 0)
            metric_dict = self.evaluate_metric(pred,target)
            # save the npy
            # heatmap_file = os.path.join(self.config.heatmap_dir, self.config.test_file + "_" + str(self.device_type) + ".npy")
            # save_npy = [
            #     {
            #         "audio_name": audio_name[i],
            #         "hdf5_path": hdf5_path[i],
            #         "index_in_hdf5": index_in_hdf5[i],
            #         "heatmap": pred_map[i],
            #         "target": target[i],
            #         "pred": pred[i]
            #     }
            #     for i in range(len(pred))
            # ]
            # np.save(heatmap_file, save_npy)

            print(self.device_type, metric_dict)
            self.log("mAP", metric_dict["mAP"], on_epoch = True, sync_dist = True, prog_bar=True)
            self.log("mAUC", metric_dict["mAUC"], on_epoch = True, sync_dist = True, prog_bar=True)
            self.log("dprime", metric_dict["dprime"], on_epoch = True, sync_dist = True, prog_bar=True)         

    def configure_optimizers(self):
        if self.config.model_type == "vit" or self.config.model_type == "swin" or self.config.model_type == 'rwkv6':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr = self.config.learning_rate, 
                betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
            )
            # Change: SWA
            # optimizer = SWA(optimizer, swa_start=10, swa_freq=5)
            def lr_foo(epoch):       
                if epoch < 3:
                    # warm up lr
                    lr_scale = self.config.lr_rate[epoch]
                    # lr_scale = 0.4 ** (3 - epoch)
                else:
                    # warmup schedule
                    lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
                    if lr_pos < -3:
                        lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
                    else:
                        lr_scale = self.config.lr_rate[lr_pos]
                    # lr_scale = self.config.lr_rate[-1] ** (bisect.bisect_left(self.config.lr_scheduler_epoch, epoch) + 1)
                    # lr_scale = 0.4 ** (bisect.bisect_left(self.config.lr_scheduler_epoch, epoch) + 2)
                    # 0.3 * (0.98 ** epoch) 
                    # # 0.4 ** (bisect.bisect_left(self.config.lr_scheduler_epoch, epoch) + 1)
                return lr_scale
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_foo
            )
            
            return [optimizer], [scheduler]
        elif self.config.model_type == "pann":
            assert False
            optimizer = optim.Adam(
                self.parameters(), lr = self.config.learning_rate, 
                betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
            )
            return optimizer
        else:
            assert False
            optimizer = optim.AdamW(
                self.parameters(), lr = self.config.learning_rate, 
                betas =(0.9, 0.999), eps = 1e-08, weight_decay = 0.05, amsgrad = True
            )
            return optimizer 



class Ensemble_SEDWrapper(pl.LightningModule):
    def __init__(self, sed_models, config, dataset):
        super().__init__()

        self.sed_models = nn.ModuleList(sed_models)
        self.config = config
        self.dataset = dataset

    def evaluate_metric(self, pred, ans):
        mAP = np.mean(average_precision_score(ans,pred, average = None))
        mAUC = np.mean(roc_auc_score(ans, pred, average = None))
        dprime = d_prime(mAUC)
        return {"mAP": mAP, "mAUC": mAUC, "dprime": dprime}
        
    def forward(self, x, sed_index, mix_lambda = None):
        output_dict_clip, output_dict_frame = self.sed_models[sed_index](x, mix_lambda)
        return output_dict_clip, output_dict_frame

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        
        pred = torch.zeros(len(batch["waveform"]), self.config.classes_num).float().to(self.device_type)
        for i in range(len(self.sed_models)):
            temp_pred, _ = self(batch["waveform"], i)
            pred = pred + temp_pred
        pred = pred / len(self.sed_models)
        return [
            pred.detach().cpu().numpy(), 
            batch["target"].detach().cpu().numpy(), 
        ]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device
        pred = np.concatenate([d[0] for d in test_step_outputs], axis = 0)
        target = np.concatenate([d[1] for d in test_step_outputs], axis = 0)
  
        metric_dict = self.evaluate_metric(pred,target)

        self.print(self.device_type, metric_dict)
        self.log("mAP", metric_dict["mAP"], on_epoch = True, sync_dist = True, prog_bar=True)
        self.log("mAUC", metric_dict["mAUC"], on_epoch = True, sync_dist = True, prog_bar=True)
        self.log("dprime", metric_dict["dprime"], on_epoch = True, sync_dist = True, prog_bar=True)         
    
