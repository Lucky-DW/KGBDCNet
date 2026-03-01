import torch
from torch import nn, einsum
import torchvision.models as models
import torch.nn.functional as F
import clip
import open_clip
from transformers import MambaConfig
from model_block import CaMambaModel, MemoryTransformer, TextSemanticFusion
# from model.model_block import CaMambaModel, MemoryTransformer, TextSemanticFusion

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=8, alpha=8, **kwargs):
        super().__init__(in_features, out_features, **kwargs)

        # 获取原始层的设备和数据类型
        device = self.weight.device
        dtype = self.weight.dtype

        # 冻结原始参数
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # 初始化LoRA参数（继承设备和类型）
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        self.scaling = alpha / r

        # 参数初始化
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 确保输入数据与参数在同一设备
        x = x.to(self.weight.device)

        # 原始线性层输出
        orig_output = F.linear(
            x,
            self.weight.to(x.dtype),  # 确保类型匹配
            self.bias.to(x.dtype) if self.bias is not None else None
        )

        # LoRA旁支输出
        lora_output = (x @ self.lora_A.T.to(x.dtype)) @ self.lora_B.T.to(x.dtype)
        return orig_output + self.scaling * lora_output

def replace_linear_with_lora(module, target_layers, r=8, alpha=8):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(layer in name for layer in target_layers):
            # 创建新层并继承设备信息
            new_layer = LoRALinear(
                child.in_features,
                child.out_features,
                r=r,
                alpha=alpha,
                bias=(child.bias is not None)
            )

            # 继承原始权重和偏置
            with torch.no_grad():
                new_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()

            # 保持设备一致性
            new_layer.to(child.weight.device)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_lora(child, target_layers, r, alpha)

def apply_lora_to_clip(model, r=8, alpha=8):
    # 定义目标层名称模式
    vision_targets = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]
    text_targets = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]

    # 处理视觉编码器
    replace_linear_with_lora(model.visual.transformer, vision_targets, r, alpha)

    # 处理文本编码器
    replace_linear_with_lora(model.transformer, text_targets, r, alpha)

    return model

def get_prompts(prompt_style='keywords'):

    # categories = ["building", "house", "roof", "wall", "structure", "frame", "interior", "exterior",
    #               "cracks", "holes", "collapse", "debris", "rubble", "ruins", "destruction", "disrepair",
    #               "scars", "bricks", "gravel", "stone", "tiles", "rock", "traces", "patches", "stains",
    #               "marks", "outlines", "spot-like", "cracked", "fire"]

    categories = ["building", "bungalow", "mansion", "villa", "warehouse", "factory", "cottage", "block", "road",
                  "street", "bypass", "lane", "crossroad", "intersection", "path", "track", "trail", "viaduct",
                  "roundabout", "forest", "meadow", "grassland", "lake", "pond", "jungle", "moorland", "wasteland",
                  "field", "land", "bridge", "overpass", "park", "playground", "stadium", "parking", "reservoir",
                  "depot", "storage", "tree", "bush", "grass", "shrubs", "vegetation", "woodland"]

    sentences = ["a photo of a building", "a photo of a damaged building", "a photo contains of the rubbles",
                 "a photo contains of the black holes",
                 "a photo contains of the stone", "a photo contains of the fire"]

    attributes = ["building with cracks", "collapsed structure with debris", "building with fire damage",
                  "traces of a former building", "spot-like stains on the ground", "ruined house with rubble"]

    if prompt_style == 'keywords' or prompt_style == 'soft_keywords':
        return categories
    elif prompt_style == 'sentences' or prompt_style == 'soft_sentences':
        return sentences
    elif prompt_style == 'attributes' or prompt_style == 'soft_attributes':
        return attributes
    elif prompt_style == 'soft':
        # 对于纯软提示，我们不需要生成文本，返回一个空列表作为占位符
        return []
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

class Encoder(nn.Module):
    def __init__(self, network, data_name='LEVIR_CC', embed_dim=512, feat_dim=768,
                 prompt_style='soft_keywords', soft_prompt_len=16):
        super(Encoder, self).__init__()
        self.network = network
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.prompt_style = prompt_style
        self.soft_prompt_len = soft_prompt_len
        if self.network == 'alexnet':
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'vgg19':
            cnn = models.vgg19(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'inception':
            cnn = models.inception_v3(pretrained=True, aux_logits=False)
            modules = list(cnn.children())[:-3]
        elif self.network == 'resnet18':
            cnn = models.resnet18(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet34':
            cnn = models.resnet34(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet50':
            cnn = models.resnet50(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet101':
            cnn = models.resnet101(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet152':
            cnn = models.resnet152(pretrained=True)
            modules = list(cnn.children())[:-2]

        if 'CLIP' in self.network:
            # 根据prompt_style获取硬提示文本
            prompts = get_prompts(self.prompt_style)
            print(f"Using prompt style: '{self.prompt_style}' with {len(prompts)} text prompts.")

            if 'Remote' in self.network:
                model_name = self.network.replace('RemoteCLIP-', '')
                self.clip_model, _, process = open_clip.create_model_and_transforms(model_name)
                tokenizer = open_clip.get_tokenizer(model_name)
                # self.clip_model.load_state_dict(torch.load(self.network + '.pt'))
                self.clip_model.load_state_dict(torch.load('model/' + self.network + '.pt'))
                self.clip_model.cuda()

            else:
                clip_model_type = self.network.replace('CLIP-', '')
                self.clip_model, preprocess = clip.load(clip_model_type, jit=False)
                self.clip_model = self.clip_model.to(dtype=torch.float32)

            # 如果是包含软提示的模式，则初始化软提示
            if 'soft' in self.prompt_style:
                print(f"Initializing soft prompt for style '{self.prompt_style}' with length: {self.soft_prompt_len}")
                prompt_dim = self.clip_model.ln_final.weight.shape[0]  # 获取CLIP文本嵌入维度
                self.soft_prompt = nn.Parameter(torch.zeros(1, self.soft_prompt_len, prompt_dim))
                nn.init.normal_(self.soft_prompt, std=0.02)
            else:
                self.soft_prompt = None

            # 如果不是纯软提示模式，则编码硬提示文本
            if self.prompt_style != 'soft':
                if 'Remote' in self.network:
                    self.text_embedding = tokenizer(prompts).cuda()
                else:
                    self.text_embedding = clip.tokenize(prompts).cuda()
            else:
                self.text_embedding = None

            self.fusion1 = TextSemanticFusion(in_dim=self.embed_dim, out_dim=self.feat_dim)
            self.fusion2 = TextSemanticFusion(in_dim=self.embed_dim, out_dim=self.feat_dim)

        else:
            self.cnn = nn.ModuleList(modules)

        self.fine_tune()

    def forward(self, imageA, imageB):
        if "CLIP" in self.network:
            img_A = imageA.to(dtype=torch.float32)
            img_B = imageB.to(dtype=torch.float32)
            clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
            clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)

            if self.prompt_style == 'soft':
                # 纯软提示，直接使用可学习的参数作为文本特征
                text_feat = self.soft_prompt.mean(dim=1)
            elif 'soft' in self.prompt_style:
                # 组合提示 (e.g., soft_keywords)，结合硬提示和软提示
                hard_feat = self.clip_model.encode_text(self.text_embedding)
                soft_feat = self.soft_prompt.mean(dim=1)
                # 将硬提示和软提示的特征相加
                text_feat = hard_feat + soft_feat
            else:
                # 纯硬提示，正常编码文本
                text_feat = self.clip_model.encode_text(self.text_embedding)

            logit_A = (clip_emb_A @ text_feat.T).softmax(dim=-1)
            logit_B = (clip_emb_B @ text_feat.T).softmax(dim=-1)

            clip_emb_A = self.fusion1(clip_emb_A, logit_A, text_feat).unsqueeze(1)
            clip_emb_B = self.fusion2(clip_emb_B, logit_B, text_feat).unsqueeze(1)

            img_feat_A = img_feat_A * clip_emb_A + img_feat_A
            img_feat_B = img_feat_B * clip_emb_B + img_feat_B

        else:
            feat1 = imageA
            feat2 = imageB
            feat1_list = []
            feat2_list = []
            cnn_list = list(self.cnn.children())
            for module in cnn_list:
                feat1 = module(feat1)
                feat2 = module(feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
            img_feat_A = feat1_list[-4:][-1]
            img_feat_B = feat2_list[-4:][-1]
            clip_emb_A = None
            clip_emb_B = None

        return [img_feat_A, clip_emb_A], [img_feat_B, clip_emb_B]

    def fine_tune(self, fine_tune=True):
        if 'CLIP' in self.network and fine_tune:
            self.clip_model = apply_lora_to_clip(self.clip_model)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            for name, param in self.clip_model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True

            if 'soft' in self.prompt_style and self.soft_prompt is not None:
                self.soft_prompt.requires_grad = True


        elif 'CLIP' not in self.network and fine_tune:
            for c in list(self.cnn.children())[:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class AttentiveEncoder(nn.Module):
    def __init__(self, n_layers, feature_size, heads, dropout=0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.n_layers = n_layers
        self.channels = channels

        self.h_embedding = nn.Embedding(h_feat, int(channels / 2))
        self.w_embedding = nn.Embedding(w_feat, int(channels / 2))

        config_1 = MambaConfig(num_hidden_layers=1, conv_kernel=3, hidden_size=channels)
        config_2 = MambaConfig(num_hidden_layers=1, conv_kernel=3, hidden_size=channels)

        self.CaMalayer_list = nn.ModuleList([])
        for i in range(n_layers):
            self.CaMalayer_list.append(nn.ModuleList([
                CaMambaModel(config_1),
                CaMambaModel(config_1),
            ]))
        self.cross_mamba1 = CaMambaModel(config_2)
        self.cross_mamba2 = CaMambaModel(config_2)
        self.linear = nn.Linear(2 * self.channels, self.channels)

        self.memory = MemoryTransformer(dropout=dropout, d_model=self.channels, n_head=heads)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos_embedding(self, x):
        if len(x.shape) == 3:  # NLD
            b = x.shape[0]
            c = x.shape[-1]
            x = x.transpose(-1, 1).view(b, c, self.h_feat, self.w_feat)
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        x = x.view(batch, c, -1).transpose(-1, 1)
        return x

    def forward(self, feat_A, feat_B):
        A_feat, A_pos = feat_A[0], feat_A[1]
        B_feat, B_pos = feat_B[0], feat_B[1]
        if A_pos == None:
            A_pos = torch.zeros(
                (A_feat.shape[0], self.channels), dtype=torch.float32
            ).unsqueeze(1).cuda()

            B_pos = torch.ones(
                (B_feat.shape[0], self.channels), dtype=torch.float32
            ).unsqueeze(1).cuda()

        img_sa1 = self.add_pos_embedding(A_feat)
        img_sa2 = self.add_pos_embedding(B_feat)

        for i in range(self.n_layers):
            img_sa1 = self.CaMalayer_list[i][0](inputs_embeds=img_sa1, inputs_embeds_2=img_sa1).last_hidden_state
            img_sa2 = self.CaMalayer_list[i][1](inputs_embeds=img_sa2, inputs_embeds_2=img_sa2).last_hidden_state

            if i == self.n_layers-1:
                dif = img_sa2 - img_sa1
                img_sa1 = self.cross_mamba1(inputs_embeds=img_sa1, inputs_embeds_2=dif).last_hidden_state
                img_sa2 = self.cross_mamba2(inputs_embeds=img_sa2, inputs_embeds_2=dif).last_hidden_state
                img_memory = self.linear(torch.cat((img_sa1, img_sa2), -1))
                feat_A = torch.cat((A_pos, img_sa1), 1)
                feat_B = torch.cat((B_pos, img_sa2), 1)
                feat_cap = self.memory(feat_A, feat_B, img_memory)

        feat_cap = feat_cap[:, 1:, :].transpose(-1, 1)

        return feat_cap


if __name__ == '__main__':
    # test
    img_A = torch.randn(16, 3, 224, 224).cuda()
    img_B = torch.randn(16, 3, 224, 224).cuda()

    # param prompt_style:  ('keywords', 'sentences', 'attributes', 'soft', 'soft_keywords', 'soft_sentences', 'soft_attributes')

    encoder = Encoder('RemoteCLIP-ViT-B-32', 'LEVIR_CC', 512, 768, prompt_style='soft_keywords', soft_prompt_len=16).cuda()
    encoder.fine_tune(True)
    attentiveencoder = AttentiveEncoder(n_layers=3, feature_size=(7, 7, 768), heads=8).cuda()
    feat_A, feat_B = encoder(img_A, img_B)
    feat_cap = attentiveencoder(feat_A, feat_B)
    print(f"\nFinal output shape: {feat_cap.shape}")
    print('Done')
