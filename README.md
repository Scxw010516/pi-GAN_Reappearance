# π-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis
### [Project Page](https://marcoamonteiro.github.io/pi-GAN-website/) | [Paper](https://arxiv.org/pdf/2012.00926.pdf) | [Data]()
[Eric Ryan Chan](https://ericryanchan.github.io/about.html)\*,
[Marco Monteiro](https://twitter.com/MonteiroAMarco)\*,
[Petr Kellnhofer](https://kellnhofer.xyz/),
[Jiajun Wu](https://jiajunwu.com/),
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)<br>
\*denotes equal contribution

This is the official implementation of the paper "π-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis".

π-GAN is a novel generative model for high-quality 3D aware image synthesis.

https://user-images.githubusercontent.com/9628319/122865841-e2d1c080-d2db-11eb-9621-1e176db59352.mp4

## Training a Model

The main training script can be found in train.py. Majority of hyperparameters for training and evaluation are set in the curriculums.py file. (see file for more details) We provide recommended curriculums for CelebA, Cats, and CARLA.

主要的训练脚本可以在train.py中找到。大多数用于训练和评估的超参数都设置在curriculums.py文件中。（有关更多详细信息，请参阅文件）我们为CelebA、Cats和CARLA提供推荐课程。
##### Relevant Flags:

Set the output directory 设置输出目录:
`--output_dir=[output directory]`

Set the model loading directory 设置模型加载目录:
`--load_dir=[load directory]`

Set the current training curriculum 设置当前的训练课程:
`--curriculum=[curriculum]`

Set the port for distributed training 设置用于分布式培训的端口:
`--port=[port]`


##### To start training:

Update the `dataset_path` field in the curriculum to point to your images.

On one GPU for CelebA:
`CUDA_VISIBLE_DEVICES=0 python3 train.py --curriculum CelebA --output_dir celebAOutputDir`

On multiple GPUs, simply list cuda visible devices in a comma-separated list:
`CUDA_VISIBLE_DEVICES=1,3 python3 train.py --curriculum CelebA --output_dir celebAOutputDir`

To continue training from another run specify the `--load_dir=path/to/directory` flag. 


更新课程中的“dataset_path”字段以指向您的图像。

在单个GPU上训练CelebA：
`CUDA_VISIBLE_DEVICES=0 python3 train.py --curriculum CelebA --output_dir celebAOutputDir`

在多个GPU上，只需在逗号分隔的列表中列出cuda可见设备：
`CUDA_VISIBLE_DEVICES=1,3 python3 train.py --curriculum CelebA --output_dir celebAOutputDir`

要从另一次运行中继续训练，请指定`--load_dir=path/to/directory`标志。
## Model Results and Evaluation模型结果和评估

#### Evaluation Metrics评估指标
To generate real images for evaluation run
`python fid_evaluation --dataset CelebA --img_size 128 --num_imgs 8000`.

To calculate fid/kid/inception scores run
`python eval_metrics.py path/to/generator.pth --real_image_dir path/to/real_images/directory --curriculum CelebA --num_images 8000`.

生成用于评估运行的真实图像
`python fid_evaluation --dataset CelebA --img_size 128 --num_imgs 8000`.

计算fid/kid/inception scores参数
`python eval_metrics.py path/to/generator.pth --real_image_dir path/to/real_images/directory --curriculum CelebA --num_images 8000`.

#### Rendering Images渲染图像
`python render_multiview_images.py path/to/generator.pth --curriculum CelebA --seeds 0 1 2 3`

For best visual results, load the EMA parameters, use truncation, increase the resolution (e.g. to 512 x 512) and increase the number of depth samples (e.g. to 24 or 36).

为了获得最佳视觉结果，加载EMA参数，使用截断，增加分辨率（例如，512 x 512），并增加深度样本的数量（例如，24或36）。
#### Rendering Videos渲染视频
`python render_video.py path/to/generator.pth --curriculum CelebA --seeds 0 1 2 3`

You can pass the flag `--lock_view_dependence` to remove view dependent effects. This can help mitigate distracting visual artifacts such as shifting eyebrows. However, locking view dependence may lower the visual quality of images (edges may be blurrier etc.)

您可以传递标志“--lock_view_dependence”来移除视图相关的效果。这有助于减轻分散注意力的视觉假象，如眉毛移位。然而，锁定视图依赖性可能会降低图像的视觉质量（边缘可能更模糊等）
#### Rendering Videos Interpolating between faces渲染视频在面之间插入
`python render_video_interpolation.py path/to/generator.pth --curriculum CelebA --seeds 0 1 2 3`

#### Extracting 3D Shapes提取三维形状

`python extract_shapes.py path/to/generator.pth --curriculum CelebA --seed 0`

## Pretrained Models预训练模型
We provide pretrained models for CelebA, Cats, and CARLA.

我们为CelebA、Cats和CARLA提供预训练模型。

CelebA: https://drive.google.com/file/d/1bRB4-KxQplJryJvqyEa8Ixkf_BVm4Nn6/view?usp=sharing

Cats: https://drive.google.com/file/d/1WBA-WI8DA7FqXn7__0TdBO0eO08C_EhG/view?usp=sharing

CARLA: https://drive.google.com/file/d/1n4eXijbSD48oJVAbAV4hgdcTbT3Yv4xO/view?usp=sharing

All zipped model files contain a generator.pth, ema.pth, and ema2.pth files. ema.pth used a decay of 0.999 and ema2.pth used a decay of 0.9999. All evaluation scripts will by default load the EMA from the file named `ema.pth` in the same directory as the generator.pth file.

所有压缩模型文件都包含generator.pth、ema.pth和ema2.pth文件。ema.pth使用0.999的衰减，ema2.pth使用0.9999的衰减。默认情况下，所有评估脚本都将从名为“EMA.pth”的文件中加载EMA，该文件与generator.pth文件位于同一目录中。
## Training Tips训练技巧

If you have the resources, increasing the number of samples (steps) per ray will dramatically increase the quality of your 3D shapes. If you're looking for good shapes, e.g. for CelebA, try increasing num_steps and moving the back plane (ray_end) to allow the model to move the background back and capture the full head.

如果你有资源，增加每条射线的采样数（步数）将极大地提高你的三维形状的质量。如果你正在寻找好的形状，例如CelebA，可以尝试增加num_steps，并移动背面平面（ray_end），让模型将背景向后移动并捕捉到完整的头部。

Training has been tested to work well on either two RTX 6000's or one RTX 8000. Training with smaller GPU's and batch sizes generally works fine, but it's also possible you'll encounter instability, especially at higher resolutions. Bubbles and artifacts that suddenly appear, or blurring in the tilted angles, are signs that training destabilized. This can usually be mitigated by training with a larger batch size or by reducing the learning rate.

经测试，训练在两台RTX 6000或一台RTX 8000上都能很好地运行。使用较小的GPU和批量大小的训练一般都能正常工作，但也有可能会遇到不稳定的情况，特别是在较高的分辨率下。突然出现的气泡和人工制品，或倾斜角度的模糊，是训练不稳定的迹象。这通常可以通过用更大的批量训练或降低学习率来缓解。

Since the original implementation we added a pose identity component to the loss. Controlled by pos_lambda in the curriculum, the pose idedntity component helps ensure generated scenes share the same canonical pose. Empirically, it seems to improve 3D models, but may introduce a minor decrease in image quality scores.

从最初的实现开始，我们在损失中加入了一个姿势识别组件。由课程中的pos_lambda控制，姿势识别组件有助于确保生成的场景共享相同的标准姿势。根据经验，它似乎可以改善三维模型，但可能会带来图像质量分数的轻微下降。

## Citation

If you find our work useful in your research, please cite:
```
@inproceedings{piGAN2021,
  title={pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis},
  author={Eric Chan and Marco Monteiro and Petr Kellnhofer and Jiajun Wu and Gordon Wetzstein},
  year={2021},
  booktitle={Proc. CVPR},
}
```