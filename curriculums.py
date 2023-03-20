"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.

为了方便复制实验，并避免传递多个命令行参数，我们实现了一个课程工具。参数可以在课程表字典中设置。

课程表。

    课程表中的数字键指定一个upsample步骤。当当前步骤与upsample步骤相匹配时，相应的dict中的值将在课程表中更新。在upsamples指定的常见课程值。
        batch_size。批量大小。
        num_steps: 沿着射线的样本数。
        img_size。生成的图像分辨率。
        batch_split。整数，在此基础上划分批次并按顺序聚合。(由于内存限制而使用)
        gen_lr。生成器的学习率。
        disc_lr: 鉴别器学习率。

    fov: 摄像机的视场
    ray_start: 摄像机射线的近端剪裁。
    ray_end。照相机光线的远端剪裁。
    fade_steps。上采样后，在判别器上淡化新层的步骤数。
    h_stddev：相机偏航的奇异度，单位为弧度。
    v_stddev：摄像机俯仰角度的变化，以弧度为单位。
    h_mean:  摄像机偏航的平均值（弧度）。
    v_mean: 摄像机偏航的平均值，以弧度为单位。
    sample_dist。摄像机姿态分布的类型。(高斯｜球面｜均匀｜均匀）。
    topk_interval。淡化顶k比例的间隔。
    topk_v。在top k训练过程中保留的最小批次的分数。
    betas。Adam的Beta参数。
    unique_lr: 是否在映射网络中使用减少的LRs。
    weight_decay。权重衰减参数。
    r1_lambda。R1正则化参数。
    latent_dim。生成器中Siren网络的潜伏密度。
    grad_clip。Grad clipping参数。
    模型。生成器中使用的Siren架构。(spatialsirenbaseline | tallsiren)
    generator。生成器类。(ImplicitGenerator3d)
    判别器。鉴别器类。(ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    数据集。训练数据集。(CelebA | Carla | Cats)
    clamp_mode。Siren密度输出的箝制函数。(relu | softplus)
    z_dist: 潜伏向量分布。(gaussian | uniform)
    hierarchical_sample: 启用NeRF算法中的分层取样标志。(使采样点的数量增加一倍)
    z_labmda: 实验性潜伏代码位置一致性损失的权重。
    pos_lambda: 实验性位置一致性损失的权重参数。
    last_back: 标志，用射线上最后的采样颜色填充背景色。
"""
#训练和评估的大部分超参数
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    # 返回下一次上采样的epoch
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 512) > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # 返回当前阶段的起始epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    # 返回当前epoch的step
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    # 返回当前学习步骤的元数据
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


CelebA = {
    0: {'batch_size': 28 * 2, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/home/ericryanchan/data/celeba/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
}

CARLA = {
    0: {'batch_size': 30, 'num_steps': 48, 'img_size': 32, 'batch_split': 1, 'gen_lr': 4e-5, 'disc_lr': 4e-4},
    int(10e3): {'batch_size': 14, 'num_steps': 48, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(55e3): {'batch_size': 10, 'num_steps': 48, 'img_size': 128, 'batch_split': 5, 'gen_lr': 10e-6, 'disc_lr': 10e-5},
    int(200e3): {},

    'dataset_path': '/home/marcorm/S-GAN/data/cats_bigger_than_128x128/*.jpg',
    'fov': 30,
    'ray_start': 0.7,
    'ray_end': 1.3,
    'fade_steps': 10000,
    'sample_dist': 'spherical_uniform',
    'h_stddev': math.pi,
    'v_stddev': math.pi/4 * 85/90,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi/4 * 85/90,
    'topk_interval': 1000,
    'topk_v': 1,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 10,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'TALLSIREN',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'Carla',
    'white_back': True,
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 0,
    'learnable_dist': False,
}

'''
  batch_size。批量大小。
  num_steps: 沿着射线的样本数。
  img_size。生成的图像分辨率。
  batch_split。整数，在此基础上划分批次并按顺序聚合。(由于内存限制而使用)
  gen_lr。生成器的学习率。
  disc_lr: 鉴别器学习率。
'''

CATS = {
    0: {'batch_size': 18, 'num_steps': 24, 'img_size': 64, 'batch_split': 9, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/content/pi-GAN_Reappearance/data/cats/CAT_00/*.jpg',
    'fov': 12,
    'ray_start': 0.8,
    'ray_end': 1.2,
    'fade_steps': 10000,
    'h_stddev': 0.5,
    'v_stddev': 0.4,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'uniform',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'StridedDiscriminator',
    'dataset': 'Cats',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
}