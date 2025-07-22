import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 禁用内存上限 # 在创建模型或执行计算之前设置
from devicemanager import DeviceManager
import torch
from diffusers import UNet2DConditionModel

class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.time = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.torch.nn.Linear(1280, dim_out),
            torch.nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )

        self.s0 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_in,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_in,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1), # 64 +2-3+1 = 64
        )

        self.s1 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_out,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_out,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.torch.nn.Conv2d(dim_in,
                                             dim_out,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)

    def forward(self, x, time):
        #x -> [1, 320, 64, 64]
        #time -> [1, 1280]

        res = x

        #[1, 1280] -> [1, 640, 1, 1]
        time = self.time(time)

        #[1, 320, 64, 64] -> [1, 640, 32, 32]
        x = self.s0(x) + time

        #维度不变
        #[1, 640, 32, 32]
        x = self.s1(x)

        #[1, 320, 64, 64] -> [1, 640, 32, 32]
        if self.res:
            res = self.res(res)

        #维度不变
        #[1, 640, 32, 32]
        x = res + x

        return x
class CrossAttention(torch.nn.Module):

    def __init__(self, dim_q, dim_kv):
        #dim_q -> 320
        #dim_kv -> 768

        super().__init__()

        self.dim_q = dim_q

        self.q = torch.nn.Linear(dim_q, dim_q, bias=False)
        self.k = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.v = torch.nn.Linear(dim_kv, dim_q, bias=False)

        self.out = torch.nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        #x -> [1, 4096, 320]
        #kv -> [1, 77, 768]

        #[1, 4096, 320] -> [1, 4096, 320]
        q = self.q(q)
        #[1, 77, 768] -> [1, 77, 320]
        k = self.k(kv)
        #[1, 77, 768] -> [1, 77, 320]
        v = self.v(kv)

        def reshape(x):
            #x -> [1, 4096, 320]
            b, lens, dim = x.shape

            #[1, 4096, 320] -> [1, 4096, 8, 40]
            x = x.reshape(b, lens, 8, dim // 8)

            #[1, 4096, 8, 40] -> [1, 8, 4096, 40]
            x = x.transpose(1, 2)

            #[1, 8, 4096, 40] -> [8, 4096, 40]
            x = x.reshape(b * 8, lens, dim // 8)

            return x

        #[1, 4096, 320] -> [8, 4096, 40]
        q = reshape(q)
        #[1, 77, 320] -> [8, 77, 40]
        k = reshape(k)
        #[1, 77, 320] -> [8, 77, 40]
        v = reshape(v)

        #[8, 4096, 40] * [8, 40, 77] -> [8, 4096, 77]
        #atten = q.bmm(k.transpose(1, 2)) * (self.dim_q // 8)**-0.5

        #从数学上是等价的,但是在实际计算时会产生很小的误差
        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
            q,
            k.transpose(1, 2),
            beta=0,
            alpha=(self.dim_q // 8)**-0.5,
        )

        atten = atten.softmax(dim=-1)

        #[8, 4096, 77] * [8, 77, 40] -> [8, 4096, 40]
        atten = atten.bmm(v)

        def reshape(x):
            #x -> [8, 4096, 40]
            b, lens, dim = x.shape

            #[8, 4096, 40] -> [1, 8, 4096, 40]
            x = x.reshape(b // 8, 8, lens, dim)

            #[1, 8, 4096, 40] -> [1, 4096, 8, 40]
            x = x.transpose(1, 2)

            #[1, 4096, 320]
            x = x.reshape(b // 8, lens, dim * 8)

            return x

        #[8, 4096, 40] -> [1, 4096, 320]
        atten = reshape(atten)

        #[1, 4096, 320] -> [1, 4096, 320]
        atten = self.out(atten)

        return atten
class Transformer(torch.nn.Module):

    def __init__(self, dim): # 320
        super().__init__()

        self.dim = dim

        #in
        self.norm_in = torch.nn.GroupNorm(num_groups=32,
                                          num_channels=dim,
                                          eps=1e-6,
                                          affine=True)
        self.cnn_in = torch.nn.Conv2d(dim,
                                      dim,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        #atten
        self.norm_atten0 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten2 = CrossAttention(dim, 768)

        #act
        self.norm_act = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = torch.nn.Linear(dim, dim * 8)
        self.act = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(dim * 4, dim)

        #out
        self.cnn_out = torch.nn.Conv2d(dim,
                                       dim,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, q, kv):
        #q -> [1, 320, 64, 64]
        #kv -> [1, 77, 768]
        b, _, h, w = q.shape
        res1 = q

        #----in----
        #维度不变
        #[1, 320, 64, 64]
        q = self.cnn_in(self.norm_in(q))

        #[1, 320, 64, 64] -> [1, 64, 64, 320] -> [1, 4096, 320]
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, self.dim)

        #----atten----
        #维度不变
        #[1, 4096, 320]
        q = self.atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        q = self.atten2(q=self.norm_atten1(q), kv=kv) + q

        #----act----
        #[1, 4096, 320]
        res2 = q

        #[1, 4096, 320] -> [1, 4096, 2560]
        q = self.fc0(self.norm_act(q))

        #1280
        d = q.shape[2] // 2

        #[1, 4096, 1280] * [1, 4096, 1280] -> [1, 4096, 1280]
        q = q[:, :, :d] * self.act(q[:, :, d:])

        #[1, 4096, 1280] -> [1, 4096, 320]
        q = self.fc1(q) + res2

        #----out----
        #[1, 4096, 320] -> [1, 64, 64, 320] -> [1, 320, 64, 64]
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()

        #维度不变
        #[1, 320, 64, 64]
        q = self.cnn_out(q) + res1

        return q
class DownBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out): # 320 640
        super().__init__()

        self.tf0 = Transformer(dim_out)
        self.res0 = Resnet(dim_in, dim_out)

        self.tf1 = Transformer(dim_out)
        self.res1 = Resnet(dim_out, dim_out)

        self.out = torch.nn.Conv2d(dim_out,
                                   dim_out,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1) # 32 + 2 -3 / 2 + 1 = 16

    def forward(self, out_vae, out_encoder, time): # torch.randn(1, 320, 32, 32), torch.randn(1, 77, 768), torch.randn(1, 1280)
        outs = []

        out_vae = self.res0(out_vae, time) # (1, 640, 32, 32) 可见time和图作用为一个噪音图
        out_vae = self.tf0(out_vae, out_encoder) # (1, 640, 32, 32) 可见文本通过tf作用噪音图
        outs.append(out_vae)

        out_vae = self.res1(out_vae, time) #(1, 640, 32, 32)
        out_vae = self.tf1(out_vae, out_encoder) # (1, 640, 32, 32)
        outs.append(out_vae)

        out_vae = self.out(out_vae) #(1, 640, 16, 16)
        outs.append(out_vae)

        return out_vae, outs

class UpBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_prev, add_up): # (320, 640, 1280, True)
        super().__init__()

        self.res0 = Resnet(dim_out + dim_prev, dim_out)
        self.res1 = Resnet(dim_out + dim_out, dim_out)
        self.res2 = Resnet(dim_in + dim_out, dim_out)

        self.tf0 = Transformer(dim_out)
        self.tf1 = Transformer(dim_out)
        self.tf2 = Transformer(dim_out)

        self.out = None
        if add_up:
            self.out = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            )

    def forward(self, out_vae, out_encoder, time, out_down):
         # (torch.randn(1, 1280, 32, 32),
         # torch.randn(1, 77, 768),
         # torch.randn(1, 1280),
         # [
         #  torch.randn(1, 320, 32, 32),
         #  torch.randn(1, 640, 32, 32),
         #  torch.randn(1, 640, 32, 32)
         # ]
        out_vae = self.res0(torch.cat([out_vae, out_down.pop()], dim=1), time) # (1, 640, 32, 32)
        out_vae = self.tf0(out_vae, out_encoder)

        out_vae = self.res1(torch.cat([out_vae, out_down.pop()], dim=1), time) # (1, 640, 32, 32)
        out_vae = self.tf1(out_vae, out_encoder)

        out_vae = self.res2(torch.cat([out_vae, out_down.pop()], dim=1), time) # (1, 640+320->640, 32, 32)  通道减半
        out_vae = self.tf2(out_vae, out_encoder)

        if self.out:
            out_vae = self.out(out_vae) #宽高加倍

        return out_vae

class UNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #in
        self.in_vae = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280),
        )

        #down
        self.down_block0 = DownBlock(320, 320)
        self.down_block1 = DownBlock(320, 640)
        self.down_block2 = DownBlock(640, 1280)

        self.down_res0 = Resnet(1280, 1280)
        self.down_res1 = Resnet(1280, 1280)

        #mid
        self.mid_res0 = Resnet(1280, 1280)
        self.mid_tf = Transformer(1280)
        self.mid_res1 = Resnet(1280, 1280)

        #up
        self.up_res0 = Resnet(2560, 1280)
        self.up_res1 = Resnet(2560, 1280)
        self.up_res2 = Resnet(2560, 1280)

        self.up_in = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
        )

        self.up_block0 = UpBlock(640, 1280, 1280, True)
        self.up_block1 = UpBlock(320, 640, 1280, True)
        self.up_block2 = UpBlock(320, 320, 640, False)

        #out
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
            torch.nn.SiLU(),
            torch.nn.Conv2d(320, 4, kernel_size=3, padding=1),
        )

    def forward(self, out_vae, out_encoder, time):
        #out_vae -> [1, 4, 64, 64]
        #out_encoder -> [1, 77, 768]
        #time -> [1]

        #----in----
        #[1, 4, 64, 64] -> [1, 320, 64, 64]
        out_vae = self.in_vae(out_vae)

        def get_time_embed(t):
            #-9.210340371976184 = -math.log(10000)
            e = torch.arange(160) * -9.210340371976184 / 160
            e = e.exp().to(t.device) * t

            #[160+160] -> [320] -> [1, 320]
            e = torch.cat([e.cos(), e.sin()]).unsqueeze(dim=0)

            return e

        #[1] -> [1, 320]
        time = get_time_embed(time)
        #[1, 320] -> [1, 1280]
        time = self.in_time(time)

        #----down----
        #[1, 320, 64, 64]
        #[1, 320, 64, 64] down_block0
        #[1, 320, 64, 64] down_block0
        #[1, 320, 32, 32] down_block0
        #[1, 640, 32, 32] down_block1
        #[1, 640, 32, 32] down_block1
        #[1, 640, 16, 16] down_block1
        #[1, 1280, 16, 16] down_block2
        #[1, 1280, 16, 16] down_block2
        #[1, 1280, 8, 8] down_block2
        #[1, 1280, 8, 8] down_res0
        #[1, 1280, 8, 8] down_res1
        out_down = [out_vae]

        #[1, 320, 64, 64],[1, 77, 768],[1, 1280] -> [1, 320, 32, 32]
        #out -> [1, 320, 64, 64],[1, 320, 64, 64][1, 320, 32, 32]
        out_vae, out = self.down_block0(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        #[1, 320, 32, 32],[1, 77, 768],[1, 1280] -> [1, 640, 16, 16]
        #out -> [1, 640, 32, 32],[1, 640, 32, 32],[1, 640, 16, 16]
        out_vae, out = self.down_block1(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        #[1, 640, 16, 16],[1, 77, 768],[1, 1280] -> [1, 1280, 8, 8]
        #out -> [1, 1280, 16, 16],[1, 1280, 16, 16],[1, 1280, 8, 8]
        out_vae, out = self.down_block2(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.down_res0(out_vae, time)
        out_down.append(out_vae)

        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.down_res1(out_vae, time)
        out_down.append(out_vae)

        #----mid----
        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.mid_res0(out_vae, time)

        #[1, 1280, 8, 8],[1, 77, 768] -> [1, 1280, 8, 8]
        out_vae = self.mid_tf(out_vae, out_encoder)

        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.mid_res1(out_vae, time)

        #----up----
        #[1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.up_res0(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)

        #[1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.up_res1(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)

        #[1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.up_res2(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)

        #[1, 1280, 8, 8] -> [1, 1280, 16, 16]
        out_vae = self.up_in(out_vae)

        #[1, 1280, 16, 16],[1, 77, 768],[1, 1280] -> [1, 1280, 32, 32]
        #out_down -> [1, 640, 16, 16],[1, 1280, 16, 16],[1, 1280, 16, 16]
        out_vae = self.up_block0(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        #[1, 1280, 32, 32],[1, 77, 768],[1, 1280] -> [1, 640, 64, 64]
        #out_down -> [1, 320, 32, 32],[1, 640, 32, 32],[1, 640, 32, 32]
        out_vae = self.up_block1(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        #[1, 640, 64, 64],[1, 77, 768],[1, 1280] -> [1, 320, 64, 64]
        #out_down -> [1, 320, 64, 64],[1, 320, 64, 64],[1, 320, 64, 64]
        out_vae = self.up_block2(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        #----out----
        #[1, 320, 64, 64] -> [1, 4, 64, 64]
        out_vae = self.out(out_vae)

        return out_vae

class LoadUnetParam:
    def load(self, device):
        print('unet加载预训练模型的参数')
        # 加载预训练模型的参数
        params = UNet2DConditionModel.from_pretrained(
            'caochongyang/diffsion_from_scratch.params', subfolder='unet',use_safetensors=False).to(device)

        unet = UNet().to(device)

        # in
        unet.in_vae.load_state_dict(params.conv_in.state_dict())
        unet.in_time[0].load_state_dict(params.time_embedding.linear_1.state_dict())
        unet.in_time[2].load_state_dict(params.time_embedding.linear_2.state_dict())

        # down
        def load_tf(model, param):
            model.norm_in.load_state_dict(param.norm.state_dict())
            model.cnn_in.load_state_dict(param.proj_in.state_dict())

            model.atten1.q.load_state_dict(
                param.transformer_blocks[0].attn1.to_q.state_dict())
            model.atten1.k.load_state_dict(
                param.transformer_blocks[0].attn1.to_k.state_dict())
            model.atten1.v.load_state_dict(
                param.transformer_blocks[0].attn1.to_v.state_dict())
            model.atten1.out.load_state_dict(
                param.transformer_blocks[0].attn1.to_out[0].state_dict())

            model.atten2.q.load_state_dict(
                param.transformer_blocks[0].attn2.to_q.state_dict())
            model.atten2.k.load_state_dict(
                param.transformer_blocks[0].attn2.to_k.state_dict())
            model.atten2.v.load_state_dict(
                param.transformer_blocks[0].attn2.to_v.state_dict())
            model.atten2.out.load_state_dict(
                param.transformer_blocks[0].attn2.to_out[0].state_dict())

            model.fc0.load_state_dict(
                param.transformer_blocks[0].ff.net[0].proj.state_dict())

            model.fc1.load_state_dict(
                param.transformer_blocks[0].ff.net[2].state_dict())

            model.norm_atten0.load_state_dict(
                param.transformer_blocks[0].norm1.state_dict())
            model.norm_atten1.load_state_dict(
                param.transformer_blocks[0].norm2.state_dict())
            model.norm_act.load_state_dict(
                param.transformer_blocks[0].norm3.state_dict())

            model.cnn_out.load_state_dict(param.proj_out.state_dict())

        def load_res(model, param):
            model.time[1].load_state_dict(param.time_emb_proj.state_dict())

            model.s0[0].load_state_dict(param.norm1.state_dict())
            model.s0[2].load_state_dict(param.conv1.state_dict())

            model.s1[0].load_state_dict(param.norm2.state_dict())
            model.s1[2].load_state_dict(param.conv2.state_dict())

            if isinstance(model.res, torch.nn.Module):
                model.res.load_state_dict(param.conv_shortcut.state_dict())

        def load_down_block(model, param):
            load_tf(model.tf0, param.attentions[0])
            load_tf(model.tf1, param.attentions[1])

            load_res(model.res0, param.resnets[0])
            load_res(model.res1, param.resnets[1])

            model.out.load_state_dict(param.downsamplers[0].conv.state_dict())

        load_down_block(unet.down_block0, params.down_blocks[0])
        load_down_block(unet.down_block1, params.down_blocks[1])
        load_down_block(unet.down_block2, params.down_blocks[2])

        load_res(unet.down_res0, params.down_blocks[3].resnets[0])
        load_res(unet.down_res1, params.down_blocks[3].resnets[1])

        # mid
        load_tf(unet.mid_tf, params.mid_block.attentions[0])
        load_res(unet.mid_res0, params.mid_block.resnets[0])
        load_res(unet.mid_res1, params.mid_block.resnets[1])

        # up
        load_res(unet.up_res0, params.up_blocks[0].resnets[0])
        load_res(unet.up_res1, params.up_blocks[0].resnets[1])
        load_res(unet.up_res2, params.up_blocks[0].resnets[2])
        unet.up_in[1].load_state_dict(
            params.up_blocks[0].upsamplers[0].conv.state_dict())

        def load_up_block(model, param):
            load_tf(model.tf0, param.attentions[0])
            load_tf(model.tf1, param.attentions[1])
            load_tf(model.tf2, param.attentions[2])

            load_res(model.res0, param.resnets[0])
            load_res(model.res1, param.resnets[1])
            load_res(model.res2, param.resnets[2])

            if isinstance(model.out, torch.nn.Module):
                model.out[1].load_state_dict(param.upsamplers[0].conv.state_dict())

        load_up_block(unet.up_block0, params.up_blocks[1])
        load_up_block(unet.up_block1, params.up_blocks[2])
        load_up_block(unet.up_block2, params.up_blocks[3])

        # out
        unet.out[0].load_state_dict(params.conv_norm_out.state_dict())
        unet.out[2].load_state_dict(params.conv_out.state_dict())

        return unet, params


if __name__ == '__main__':
    # print(Resnet(320, 640)(torch.randn(1, 320, 32, 32), torch.randn(1, 1280)).shape) # torch.Size([1, 640, 32, 32])
    # print(CrossAttention(320, 768)(torch.randn(1, 4096, 320), torch.randn(1, 77,768)).shape) # torch.Size([1, 4096, 320])
    # print(Transformer(320)(torch.randn(1, 320, 64, 64), torch.randn(1, 77, 768)).shape) # torch.Size([1, 320, 64, 64])
    # print(DownBlock(320, 640)(torch.randn(1, 320, 32, 32), torch.randn(1, 77, 768),torch.randn(1, 1280))[0].shape) # torch.Size([1, 640, 16, 16])
    # print(UpBlock(320, 640, 1280, True)(torch.randn(1, 1280, 32, 32),
    #                           torch.randn(1, 77, 768), torch.randn(1, 1280), [
    #                               torch.randn(1, 320, 32, 32),
    #                               torch.randn(1, 640, 32, 32),
    #                               torch.randn(1, 640, 32, 32)
    #                           ]).shape) # torch.Size([1, 640, 64, 64])
    # print(UNet()(torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768),torch.LongTensor([26])).shape) # torch.Size([2, 4, 64, 64])

    #### 验证 ####
    device_manager = DeviceManager()
    device =  device_manager.get_device() #torch.device("cpu") #
    data = torch.randn(1, 3, 512, 512, device=device)
    unet, params = LoadUnetParam().load(device)

    out_vae = torch.randn(1, 4, 64, 64,device=device)
    out_encoder = torch.randn(1, 77, 768,device=device)
    time = torch.LongTensor([26]).to(device) # 随机一个0到999的强度

    a = unet(out_vae=out_vae, out_encoder=out_encoder, time=time)
    b = params(out_vae, time, out_encoder).sample
    print(a[0][2])
    print(b[0][2])
    print((a == b).all())

