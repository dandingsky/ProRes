# Progressive Residual Warmup

This repository provides reproduction code for our paper [Progressive Residual Warmup for Language Model Pretraining](https://arxiv.org/abs/2603.05369).

## Setup
You can run the following commands to prepare the conda environment:
```bash
conda create -n prores python=3.9 -y
conda activate prores
pip install -r requirements.txt
```
Before running the training scripts, you might need to login to HuggingFace and WandB using these commands:
```bash
huggingface-cli login
wandb login
```
The first enables streaming the C4 dataset during training, and the second is used for tracking training statistics.
For longer runs such as the 1B and 7B parameter experiments, we recommend downloading the C4 dataset locally to avoid streaming error.

> [!NOTE]
> In the paper, sequences were packed to a length of 1024 before training. 
> For simplicity, sequence packing is omitted in this repository.

## Reproduce Pretraining Results

### Main experiments
Run `scripts/run_1b.sh`, `scripts/run_350m.sh`, or `scripts/run_130m.sh`. For example:
```bash
sh scripts/run_1b.sh pre
sh scripts/run_1b.sh pre_prores
```
Here `pre` and `pre_prores` refer to the baseline types Pre-LN (without ProRes) and Pre-LN (with ProRes), respectively.
The following types are supported:

| Method             | without ProRes            | with ProRes                     |
|--------------------|---------------------------|---------------------------------|
| Pre-LN             | `pre`                     | `pre_prores`                    |
| Sandwich-LN        | `sandwich`                | `sandwich_prores`               |
| LayerNorm Scaling  | `lns`                     | `lns_prores`                    |
| Post-LN            | `post`                    | `post_prores`                   |
| DeepNorm           | `deeppost`                | `deeppost_prores`               |

> [!NOTE]
> The training scripts are configured for a node with 8×H800 GPUs (80GB each).  
> You may need to adjust parameters such as `--batch_size` to match your hardware.


### Depth Scaling Experiments
Use the scripts in `scripts/depth_scaling/`, such as:
```bash
sh scripts/depth_scaling/run_71m_120L.sh pre
```

### Warmup Schedule Ablation
You can configure the type of ProRes schedule by exporting the following environment variable:
```bash
export PRORES_SCHEDULE=linear # default
```
The following schedules are implemented:
| Schedule | `PRORES_SCHEDULE` |
|---|---|
| linear | `linear` |
| linear-square | `linear_square` |
| linear-sqrt | `linear_sqrt` |
| equal | `linear_equal` |
| reverse | `linear_reverse` |
| stagewise-0 | `stagewise_0` |
| stagewise-$\sqrt{L}$ | `stagewise_sqrt_L` |
| stagewise-$\sqrt{l}$ | `stagewise_sqrt_l` |
| fix-$L$ | `fix_L` |
| fix-$\sqrt{L}$ | `fix_sqrt_L` |
| fix-$\sqrt{l}$ | `fix_sqrt_l` |

To adjust the residual warmup length, configure the `--prores_warmup_steps` option in the training script.
Note that this argument refers to the total length of the residual warmup phase, not the $T$ parameter in $\alpha(l,t)$.
For instance, to implement $T=1000$ under the "linear" schedule with model depth $L=24$, you need `--prores_warmup_steps 24000`, which corresponds to a warmup length of $T\times L$.

### Resume Training
Our training script supports checkpointing and resuming, which is useful for longer runs. You may use `--save_every` and `--continue_from` when necessary.
To resume training from a previously saved checkpoint, you can add the following arguments to the training script:
```bash
--continue_from /path/to/checkpoint \
--new_wandb_on_resume
```

The argument `--new_wandb_on_resume` is optional. By default, we resume the training process as well as the WandB run. Using a new WandB run for the resumed training can be helpful for ensuring consistent dataloader states.


## Citation
If you find this work useful, please consider citing our paper:
```
@article{chen2026prores,
      title={Progressive Residual Warmup for Language Model Pretraining}, 
      author={Tianhao Chen and Xin Xu and Lu Yin and Hao Chen and Yang Wang and Shizhe Diao and Can Yang},
      journal={arXiv preprint arXiv:2603.05369},
      year={2026}
}
```