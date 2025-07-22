import torch
from transformers import CLIPTextModel
from devicemanager import DeviceManager

class Embed(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = torch.nn.Embedding(49408, 768) # num_embeddings表示嵌入字典的大小，即词汇表的大小。也就是说，一共有49408个不同的词（或标记），每个词对应一个唯一的整数索引（从0到49407）。因此，这个嵌入层可以处理的最大索引是49407。
        self.pos_embed = torch.nn.Embedding(77, 768)

        self.register_buffer('pos_ids', torch.arange(77).unsqueeze(dim=0)) # torch.Size([1, 77])

    def forward(self, input_ids):
        #input_ids -> [b, 77]

        #[b, 77] -> [b, 77, 768]
        embed = self.embed(input_ids)
        # print(embed[0])

        #[1, 77] -> [1, 77, 768]
        pos_embed = self.pos_embed(self.pos_ids)
        # print(pos_embed[0])

        #[b, 77, 768]
        return embed + pos_embed

class Atten(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(768, 768)
        self.k = torch.nn.Linear(768, 768)
        self.v = torch.nn.Linear(768, 768)
        self.out = torch.nn.Linear(768, 768)

    def forward(self, x):
        #x -> [b, 77, 768]

        b = x.shape[0]

        #维度不变
        #[b, 77, 768]
        q = self.q(x) * 0.125
        k = self.k(x)
        v = self.v(x)

        #拆分注意力头
        #[b, 77, 768] -> [b, 77, 12, 64] -> [b, 12, 77, 64] -> [b*12, 77, 64]
        q = q.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        k = k.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        v = v.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)

        #计算qk乘积
        #[b*12, 77, 64] * [b*12, 64, 77] -> [b*12, 77, 77]
        attn = torch.bmm(q, k.transpose(1, 2))

        #[b*12, 77, 77] -> [b, 12, 77, 77]
        attn = attn.reshape(b, 12, 77, 77)

        #覆盖mask
        def get_mask(b):
            mask = torch.empty(b, 77, 77)

            #上三角的部分置为负无穷
            mask.fill_(-float('inf'))

            #对角线和以下的位置为0
            mask.triu_(1)

            return mask.unsqueeze(1)

        #[b, 12, 77, 77] + [b, 1, 77, 77] -> [b, 12, 77, 77]
        attn = attn + get_mask(attn.shape[0]).to(attn.device)

        #[b, 12, 77, 77] -> [b*12, 77, 77]
        attn = attn.reshape(b * 12, 77, 77)

        #计算softmax,被mask的部分值为0
        attn = attn.softmax(dim=-1)

        #计算和v的乘积
        #[b*12, 77, 77] * [b*12, 77, 64] -> [b*12, 77, 64]
        attn = torch.bmm(attn, v)

        #[b*12, 77, 64] -> [b, 12, 77, 64] -> [b, 77, 12, 64] -> [b, 77, 768]
        attn = attn.reshape(b, 12, 77, 64).transpose(1, 2).reshape(b, 77, 768)

        #线性输出,维度不变
        #[b, 77, 768]
        return self.out(attn)

class ClipEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.s1 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            Atten(),
        )

        self.s2 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 3072),
        )

        self.s3 = torch.nn.Linear(3072, 768)

    def forward(self, x):
        #x -> [2, 77, 768]

        #维度不变
        #[2, 77, 768]
        x = x + self.s1(x)

        #[2, 77, 768]
        res = x

        #[2, 77, 768] -> [2, 77, 3072]
        x = self.s2(x)

        #维度不变
        #[2, 77, 3072]
        x = x * (x * 1.702).sigmoid()

        #[2, 77, 3072] -> [2, 77, 768]
        return res + self.s3(x)
class LoadTextEncoderParam:
    def load(self, device):
        print('text_encoder加载预训练模型的参数')
        #### 定义encoder  ####
        encoder = torch.nn.Sequential(
            Embed(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            torch.nn.LayerNorm(768),
        ).to(device)

        ####  加载预训练模型的参数 ####
        params = CLIPTextModel.from_pretrained(
            'caochongyang/diffsion_from_scratch.params', subfolder='text_encoder').to(device)

        # 词编码
        encoder[0].embed.load_state_dict(
            params.text_model.embeddings.token_embedding.state_dict())

        # 位置编码
        encoder[0].pos_embed.load_state_dict(
            params.text_model.embeddings.position_embedding.state_dict())

        # 12层编码层
        for i in range(12):
            # 第一层norm
            encoder[i + 1].s1[0].load_state_dict(
                params.text_model.encoder.layers[i].layer_norm1.state_dict())

            # 注意力q矩阵
            encoder[i + 1].s1[1].q.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.q_proj.state_dict())

            # 注意力k矩阵
            encoder[i + 1].s1[1].k.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.k_proj.state_dict())

            # 注意力v矩阵
            encoder[i + 1].s1[1].v.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.v_proj.state_dict())

            # 注意力out
            encoder[i + 1].s1[1].out.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.out_proj.state_dict())

            # 第二层norm
            encoder[i + 1].s2[0].load_state_dict(
                params.text_model.encoder.layers[i].layer_norm2.state_dict())

            # mlp第一层fc
            encoder[i + 1].s2[1].load_state_dict(
                params.text_model.encoder.layers[i].mlp.fc1.state_dict())

            # mlp第二层fc
            encoder[i + 1].s3.load_state_dict(
                params.text_model.encoder.layers[i].mlp.fc2.state_dict())

        # 输出norm
        encoder[13].load_state_dict(params.text_model.final_layer_norm.state_dict())

        return encoder, params


if __name__ == '__main__':

    # print(Embed()(torch.ones(2, 77).long()).shape) # 默认是float32 变为int64 通过tensor.dtype可看  输出纬度torch.Size([2, 77, 768])
    # print(Atten()(torch.randn(2, 77, 768)).shape)
    # print(ClipEncoder()(torch.randn(2, 77, 768)).shape)


    #### 验证 ####
    device_manager = DeviceManager() # 感觉 77 的 话 cpu 和 mps 查不了多少
    device = device_manager.get_device()
    encoder, params = LoadTextEncoderParam().load(device)
    inputData = torch.arange(77, device=device).unsqueeze(dim=0)
    # encoder.to(device)
    # params.to(device)
    # print(encoder)
    # print(params)

    a = encoder(inputData)
    b = params(inputData).last_hidden_state # last_hidden_state 是 Transformer 模型的最终层输出的数据

    print(a)
    print(b)
    print(a.shape) # torch.Size([1, 77, 768])
    print(b.shape)
    print(a.dtype)
    print(b.dtype)
    print(a[0][1])
    print(b[0][1])
    print((a == b))
    print((a == b).all()) #tensor(False, device='mps:0') 可能是系统浮点数比较有差别 看数是都都相等的
