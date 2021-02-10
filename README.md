# Computer Vision Lab 2020 Winter Research Participation 
- Course Name: CSED399
- 윤병준 20190766
- wandb project link 
    - [resnet](https://wandb.ai/happyhappy/cvlab-cifar100-efficient-net)
    - [EfficientNet-B0](https://wandb.ai/happyhappy/cvlab-cifar100-project)

## Feature

- TensorBoardX / [wandb](https://www.wandb.com/) support
- Background generator is used ([reason of using background generator](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5))
  - In Windows, background generator could not be supported. So if error occurs, set false to `use_background_generator` in config
- Training state and network checkpoint saving, loading
    - Training state includes not only network weights, but also optimizer, step, epoch.
    - Checkpoint includes only network weights. This could be used for inference. 
- [Hydra](https://hydra.cc) and [Omegaconf](https://github.com/omry/omegaconf) is supported
- Distributed Learning using Distributed Data Parallel is supported
- Config with yaml file / easy dot-style access to config
- Code lint / CI
- Code Testing with pytest

## Code Structure

- `config` dir: folder for config files
- `dataset` dir: dataloader and dataset codes are here. Also, put dataset in `meta` dir.
- `model` dir: `model.py` is for wrapping network architecture. `model_arch.py` is for coding network architecture.
- `optimizer` dir: load different optimizer and learning rate scheduler based on `config`
- `test` dir: folder for `pytest` testing codes. You can check your network's flow of tensor by fixing `tests/model/net_arch_test.py`. 
Just copy & paste `Net_arch.forward` method to  `net_arch_test.py` and add `assert` phrase to check tensor.
- `utils` dir:
    - `train_model.py` and `test_model.py` are for train and test model once.
    - `utils.py` is for utility. random seed setting, dot-access hyper parameter, get commit hash, etc are here. 
    - `writer.py` is for writing logs in tensorboard / wandb.
- `trainer.py` file: this is for setting up and iterating epoch.

## Setup

### Install requirements

- python3 (3.6, 3.7, 3.8 is tested)
- Write PyTorch version which you want to `requirements.txt`. (https://pytorch.org/get-started/)
- `pip install -r requirements.txt`

### Config

- Config is written in yaml file
    - You can choose configs at `config/default.yaml`. Custom configs are under `config/job/`
- `name` is train name you run.
- `working_dir` is root directory for saving checkpoints, logging logs.
- `device` is device mode for running your model. You can choose `cpu` or `cuda`
- `num_epoch` and `num_step` defines how long the training runs
- `data` field
    - Configs for Dataloader.
    - glob `train_dir` / `test_dir` with `file_format` for Dataloader.
    - If `divide_dataset_per_gpu` is true, origin dataset is divide into sub dataset for each gpu. 
    This could mean the size of origin dataset should be multiple of number of using gpu.
    If this option is false, dataset is not divided but epoch goes up in multiple of number of gpus.
    - `transform` field is config for transformation. It is not implemented yet to automatically change a transformer, but for logging purpose. If you want to change `transform` strategy, it is available in `dataset/dataloader.py`. (RandAugmentation, cutoff)
- `train`/`test` field
    - Configs for training options.
    - `random_seed` is for setting python, numpy, pytorch random seed.
    - `optimizer` is for selecting optimizer. Only `adam optimizer` is supported for now.
    - `dist` is for configuring Distributed Data Parallel.
        - `gpus` is the number that you want to use with DDP (`gpus` value is used at `world_size` in DDP).
        Not using DDP when `gpus` is 0, using all gpus when `gpus` is -1.
        - `timeout` is seconds for timeout of process interaction in DDP.
        When this is set as `~`, default timeout (1800 seconds) is applied in `gloo` mode and timeout is turned off in `nccl` mode.
- `model` field
    - Configs for Network architecture and options for model.
    - You can add configs in yaml format to config your network.
    - `type` is a config for model architecture
    - `optimizer` is a config for model's optimizer
    - `$(type)` field takes detailed configuration for each model structure. 
- `sweep` field
    - [ ] TODO
- `log` field
    - Configs for logging include tensorboard / wandb logging. 
    - `summary_interval` and `checkpoint_interval` are interval of step and epoch between training logging and checkpoint saving.
    - checkpoint and logs are saved under `working_dir/chkpt_dir` and `working_dir/trainer.log`. Tensorboard logs are saving under `working_dir/outputs/tensorboard`
- `load` field
    - loading from wandb server is supported
    - `wandb_load_path` is `Run path` in overview of run. If you don't want to use wandb load, this field should be `~`.
    - `network_chkpt_path` is path to network checkpoint file.
    If using wandb loading, this field should be checkpoint file name of wandb run.
    - `resume_state_path` is path to training state file.
    If using wandb loading, this field should be training state file name of wandb run.

### Code lint

1. `pip install -r requirements-dev.txt` for install develop dependencies (this requires python 3.6 and above because of black)

1. `pre-commit install` for adding pre-commit to git hook

## Train

```bash
python trainer.py
```

## Inspired by

- https://github.com/open-mmlab/mmsr
- https://github.com/allenai/allennlp (test case writing)

## Reference
```bibtex
@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{Kornblith_2019,
   title={Do Better ImageNet Models Transfer Better?},
   ISBN={9781728132938},
   url={http://dx.doi.org/10.1109/CVPR.2019.00277},
   DOI={10.1109/cvpr.2019.00277},
   journal={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   publisher={IEEE},
   author={Kornblith, Simon and Shlens, Jonathon and Le, Quoc V.},
   year={2019},
   month={Jun}
}

@misc{foret2020sharpnessaware,
    title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
    author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
    year={2020},
    eprint={2010.01412},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@misc{cubuk2018autoaugment,
    title={AutoAugment: Learning Augmentation Policies from Data},
    author={Ekin D. Cubuk and Barret Zoph and Dandelion Mane and Vijay Vasudevan and Quoc V. Le},
    year={2018},
    eprint={1805.09501},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{cubuk2019randaugment,
    title={RandAugment: Practical automated data augmentation with a reduced search space},
    author={Ekin D. Cubuk and Barret Zoph and Jonathon Shlens and Quoc V. Le},
    year={2019},
    eprint={1909.13719},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}


@misc{tan2019efficientnet,
    title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
    author={Mingxing Tan and Quoc V. Le},
    year={2019},
    eprint={1905.11946},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

