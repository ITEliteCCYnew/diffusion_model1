import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 禁用内存上限 # 在创建模型或执行计算之前设置
from devicemanager import DeviceManager
from diffusers import DiffusionPipeline
import torch
from text_encoder import LoadTextEncoderParam
from vae import LoadVAEParam
from unet import LoadUnetParam
from transformers import PreTrainedModel, PretrainedConfig
from matplotlib import pyplot as plt
from datetime import datetime


@torch.no_grad()
def generate(text):
    #词编码
    #[1, 77]
    pos = tokenizer(text,
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)
    neg = tokenizer('',
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)

    #[1, 77, 768]
    pos = encoder(pos)
    neg = encoder(neg)

    #[1+1, 77, 768] -> [2, 77, 768]
    out_encoder = torch.cat((neg, pos), dim=0)

    #vae的压缩图,从随机噪声开始
    out_vae = torch.randn(1, 4, 64, 64, device=device)

    #生成50个时间步,一般是从980-0
    scheduler.set_timesteps(50, device=device)
    for time in scheduler.timesteps:

        #往图中加噪音
        #[1+1, 4, 64, 64] -> [2, 4, 64, 64]
        noise = torch.cat((out_vae, out_vae), dim=0)
        noise = scheduler.scale_model_input(noise, time)

        #计算噪音
        #[2, 4, 64, 64],[2, 77, 768],scala -> [2, 4, 64, 64]
        pred_noise = unet(out_vae=noise, out_encoder=out_encoder, time=time)

        #从正例图中减去反例图
        #[2, 4, 64, 64] -> [1, 4, 64, 64]
        pred_noise = pred_noise[0] + 7.5 * (pred_noise[1] - pred_noise[0])

        #重新添加噪音,以进行下一步计算
        #[1, 4, 64, 64]
        out_vae = scheduler.step(pred_noise, time, out_vae).prev_sample

    #从压缩图恢复成图片
    out_vae = 1 / 0.18215 * out_vae
    #[1, 4, 64, 64] -> [1, 3, 512, 512]
    image = vae.decoder(out_vae)

    #转换成图片数据
    image = image.cpu()
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(0, 2, 3, 1)
    return image.numpy()[0]

def show():
    texts = [
        'a drawing of a star with a jewel in the center',  #宝石海星
        # 'a drawing of a woman in a red cape',  #迷唇姐
        # 'a drawing of a dragon sitting on its hind legs',  #肥大
        # 'a drawing of a blue sea turtle holding a rock',  #拉普拉斯
        # 'a blue and white bird with its wings spread',  #急冻鸟
        # 'a blue and white stuffed animal sitting on top of a white surface',  #卡比兽
    ]

    images = [generate(i) for i in texts]

    plt.figure(figsize=(10, 5))

    for i, (image, text) in enumerate(zip(images, texts)):
    # for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(text[:200] + '...' if len(text) > 20 else text, fontsize=8)
        plt.axis('off')

    # plt.savefig('pokemon_generated.png')  # 可选：保存图像
    plt.show()

#包装类
class Model(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.unet = unet.to('cpu')



if __name__ == '__main__':
    ###################
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
    # 加载模型
    encoder, _ = LoadTextEncoderParam().load(device)
    vae, _ = LoadVAEParam().load(device)
    unet, _ = LoadUnetParam().load(device)

    # 准备训练
    encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    encoder.eval()
    vae.eval()
    unet.eval()
    print("格式化时间1:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # img = generate('a drawing of a star with a jewel in the center')
    # print(img.shape)

    # 加载训练好的模型 下面已经下载到本地后从本地加载的
    local_dir = "../huggingface/caochongyang_diffsion_from_scratch_unet"
    unet = Model.from_pretrained(local_dir).unet
    # unet = Model.from_pretrained('caochongyang/diffsion_from_scratch.unet').unet
    print("格式化时间2:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    unet.eval().to(device)
    print("unet加载训练好的模型的参数:",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    show() # 50部迭代浪费时间
    print("格式化时间3:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


