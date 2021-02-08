import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import get_logger, is_logging_process
from optimizer.optimizer import *
from optimizer.lr_scheduler import *

class Model:
    def __init__(self, cfg, net_arch, loss_f, rank=0):
        self.cfg = cfg
        self.device = self.cfg.device
        self.net = net_arch.to(self.device)
        self.rank = rank
        if self.device != "cpu" and self.cfg.dist.gpus != 0:
            self.net = DDP(self.net, device_ids=[self.rank])
        self.input = None
        self.GT = None
        self.step = 0
        self.epoch = -1
        self._logger = get_logger(cfg, os.path.basename(__file__))

        self.optimizer = get_optimizer(cfg, self.net)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        
        # init loss
        self.loss_f = loss_f
        self.log = OmegaConf.create()

    def feed_data(self, **data):  # data's keys: input, GT
        for k, v in data.items():
            data[k] = v.to(self.device)
        self.input = data.get("input")
        self.GT = data.get("GT")

    def optimize_parameters(self):
        if self.cfg.model.optimizer != "SAM":
            self.net.train()
            self.optimizer.zero_grad()
            output = self.run_network()
            loss_v = self.loss_f(output, self.GT)
            loss_v.backward()
            self.optimizer.step()
            # set log
            self.log.loss_v = loss_v.item()
        else:
            self.net.train()
            self.optimizer.zero_grad()

            def closure():
                self.optimizer.zero_grad()
                output = self.run_network()
                loss_v = self.loss_f(output, self.GT)
                loss_v.backward()
                return loss_v
            loss_v = self.optimizer.step(closure)
            self.log.loss_v = loss_v.item()


    def inference(self):
        self.net.eval()
        output = self.run_network()
        return output

    def run_network(self):
        output = self.net(self.input)
        return output

    def save_network(self, save_file=True):
        if is_logging_process():
            net = self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.to("cpu")
            if save_file:
                save_filename = "%s_%d.pt" % (self.cfg.name, self.step)
                save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
                torch.save(state_dict, save_path)
                if self.cfg.log.use_wandb:
                    wandb.save(save_path)
                if is_logging_process():
                    self._logger.info("Saved network checkpoint to: %s" % save_path)
            return state_dict

    def load_network(self, loaded_net=None):
        add_log = False
        if loaded_net is None:
            add_log = True
            if self.cfg.load.wandb_load_path is not None:
                self.cfg.load.network_chkpt_path = wandb.restore(
                    self.cfg.load.network_chkpt_path,
                    run_path=self.cfg.load.wandb_load_path,
                ).name
            loaded_net = torch.load(
                self.cfg.load.network_chkpt_path,
                map_location=torch.device(self.device),
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        self.net.load_state_dict(loaded_clean_net, strict=self.cfg.load.strict_load)
        if is_logging_process() and add_log:
            self._logger.info(
                "Checkpoint %s is loaded" % self.cfg.load.network_chkpt_path
            )

    def save_training_state(self):
        if is_logging_process():
            save_filename = "%s_%d.state" % (self.cfg.name, self.step)
            save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
            net_state_dict = self.save_network(False)
            state = {
                "model": net_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            }
            torch.save(state, save_path)
            if self.cfg.log.use_wandb:
                wandb.save(save_path)
            if is_logging_process():
                self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        if self.cfg.load.wandb_load_path is not None:
            self.cfg.load.resume_state_path = wandb.restore(
                self.cfg.load.resume_state_path,
                run_path=self.cfg.load.wandb_load_path,
            ).name
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.device),
        )

        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        if is_logging_process():
            self._logger.info(
                "Resuming from training state: %s" % self.cfg.load.resume_state_path
            )
