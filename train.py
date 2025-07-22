import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 禁用内存上限 # 在创建模型或执行计算之前设置
from devicemanager import DeviceManager
from diffusers import DiffusionPipeline
import torch
from datasets import load_dataset
import torchvision
from text_encoder import LoadTextEncoderParam
from vae import LoadVAEParam
from unet import LoadUnetParam
from transformers import PreTrainedModel, PretrainedConfig

def get_loss(data):
    with torch.no_grad():
        #文字编码
        #[1, 77] -> [1, 77, 768]
        out_encoder = encoder(data['input_ids'])

        #抽取图像特征图
        #[1, 3, 512, 512] -> [1, 4, 64, 64]
        out_vae = vae.encoder(data['pixel_values'])
        out_vae = vae.sample(out_vae)

        #0.18215 = vae.config.scaling_factor
        out_vae = out_vae * 0.18215

    #随机数,unet的计算目标
    noise = torch.randn_like(out_vae)

    #往特征图中添加噪声
    #1000 = scheduler.num_train_timesteps
    #1 = batch size
    noise_step = torch.randint(0, 1000, (1, )).long().to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    #根据文字信息,把特征图中的噪声计算出来
    out_unet = unet(out_vae=out_vae_noise,
                    out_encoder=out_encoder,
                    time=noise_step)

    #计算mse loss
    #[1, 4, 64, 64],[1, 4, 64, 64]
    return criterion(out_unet, noise)

def train():
    loss_sum = 0
    for epoch in range(400):
        for i, data in enumerate(loader):
            loss = get_loss(data) / 4
            loss.backward()
            loss_sum += loss.item()

            if (epoch * len(loader) + i) % 4 == 0: #每四次进行一次参数调整
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0) #梯度裁剪，防止梯度爆炸，在深层网络（如UNet）中，反向传播时梯度可能指数级增长，裁剪可以稳定训练、加速收敛、避免NaN
                optimizer.step()
                optimizer.zero_grad()

        if epoch % 1 == 0:
            print(epoch, loss_sum)
            loss_sum = 0

    #torch.save(unet.to('cpu'), 'saves/unet.model')
    print('save local')


# 包装类
class Model(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.unet = unet.to('cpu')


if __name__ == '__main__':

    ####################
    # 加载工具类 不属于模型的一部分
    # scheduler：往图片中加入噪音的工具类（step控制噪音量 0-999之间 越大越模糊）
    # tokenizer 文本变数字。 此外embedding是数字变向量
    device = DeviceManager().get_device()

    pipeline = DiffusionPipeline.from_pretrained(
        'caochongyang/diffsion_from_scratch.params', safety_checker=None)

    scheduler = pipeline.scheduler
    tokenizer = pipeline.tokenizer

    del pipeline

    # print(device, scheduler, tokenizer)

    ####################
    # 加载数据集  pokemon数据集card 图片和文本一一对应 共833对 在线看https://huggingface.co/datasets/caochongyang/diffsion_from_scratch
    dataset = load_dataset(path='caochongyang/diffsion_from_scratch', split='train')

    # 图像增强模块
    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(512),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5]),
    ])


    def f(data):
        # 应用图像增强
        pixel_values = [compose(i) for i in data['image']]
        # 文字编码
        input_ids = tokenizer.batch_encode_plus(data['text'],
                                                padding='max_length',
                                                truncation=True,
                                                max_length=77).input_ids

        return {'pixel_values': pixel_values, 'input_ids': input_ids} # 最终得到的pixel_values的shape是3，512，512   input_ids的shape是1，77


    dataset = dataset.map(f,
                          batched=True,
                          batch_size=100, # 每次读取100个 833个读取9次
                          num_proc=1,
                          remove_columns=['image', 'text'])

    dataset.set_format(type='torch')

    # print(dataset, dataset[0]['input_ids']) #    Dataset({ features: ['pixel_values', 'input_ids'],num_rows: 833})
    # print(dataset, dataset[0]['pixel_values']) # dataset[0]['pixel_values'].shape : torch.Size([3, 512, 512])

    ####################
    # 定义loader
    def collate_fn(data):
        pixel_values = [i['pixel_values'] for i in data]
        input_ids = [i['input_ids'] for i in data]

        pixel_values = torch.stack(pixel_values).to(device)
        input_ids = torch.stack(input_ids).to(device)

        return {'pixel_values': pixel_values, 'input_ids': input_ids}
    # 执行load
    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=True,
                                         collate_fn=collate_fn,
                                         batch_size=1)

    # print(len(loader), next(iter(loader)), next(iter(loader))['pixel_values'].shape, next(iter(loader))['input_ids'].shape) # torch.Size([1, 3, 512, 512] torch.Size([1, 77])

    ####################
    # 加载模型
    encoder, _ = LoadTextEncoderParam().load(device)
    vae, _ = LoadVAEParam().load(device)
    unet, _ = LoadUnetParam().load(device)
    # 准备训练
    encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(True)

    encoder.eval()
    vae.eval()
    unet.train()

    encoder.to(device)
    vae.to(device)
    unet.to(device)

    optimizer = torch.optim.AdamW(unet.parameters(),
                                  lr=1e-5,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01,
                                  eps=1e-8)

    criterion = torch.nn.MSELoss()
    # print(get_loss({
    #     'input_ids': torch.ones(1, 77, device=device).long(),
    #     'pixel_values': torch.randn(1, 3, 512, 512, device=device)
    # })) # tensor(0.5888, device='mps:0', grad_fn=<MseLossBackward0>)

    # print(optimizer, criterion)


    ####################
    # 训练模型
    train()


    ####################
    # 保存到hub
    # Model(PretrainedConfig()).push_to_hub(
    #     repo_id='caochongyang/diffsion_from_scratch.unet',
    #     use_auth_token=open('/root/hub_token.txt').read().strip())
