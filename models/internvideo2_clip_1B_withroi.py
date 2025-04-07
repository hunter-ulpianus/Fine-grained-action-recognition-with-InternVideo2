import logging
import re
import torch
from torch import nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# from .backbones.internvideo2 import InternVideo2, LLaMA, Tokenizer
from .backbones.internvideo2 import InternVideo2_1B_branch, LLaMA, Tokenizer
from .criterions import VTC_VTM_Loss

logger = logging.getLogger(__name__)


class InternVideo2_1Broi_branch(nn.Module):
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain

        # create modules.
        if tokenizer is None:
            self.tokenizer = Tokenizer(config.model.tokenizer_path)
        self.vision_encoder = self.build_vision_encoder()      
        self.text_encoder = self.build_text_encoder()
        # adopt 1 / 100. as in ViCLIP
        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.temp_min = config.model.temp_min
        
        # freeze model
        if self.config.model.freeze_vision:
            for name, p in self.vision_encoder.named_parameters():
                # if self.config.model.open_vision_clip_projector and name.startswith('clip_projector'):
                if self.config.model.open_vision_clip_projector and "clip_projector" in name:
                    logger.info(f"Unfreeze {name}")
                else:
                    logger.info(f"Freeze {name}")
                    p.requires_grad = False
                if not config.stage2_train:     # 解锁分支
                    if "roi" in name or "global_attn" in name or "alpha" in name or "multi_scale" in name or "adapter" in name or "fusion" in name or "high" in name or "branch" in name:
                        if "blocks" in  name:
                            match = re.search(r'blocks\.(\d+)\.', name)
                            match_num = int(match.group(1))
                            if match_num > 20:
                                p.requires_grad = True
                                logger.info(f"Train new-added modul {name}")
                            else:
                                p.requires_grad = False
                                logger.info(f"Freeze {name} for blocks_idx < 30")
                        else:
                            p.requires_grad = True
                            logger.info(f"Train new-added module {name}")
                elif config.stage2_train:
                    if "blocks" in  name and not "roi" in name:
                        p.requires_grad = True
                        logger.info(f"Tune main block {name}")

                    test = None

        if self.config.model.freeze_text:
            for name, p in self.text_encoder.named_parameters():
                if self.config.model.open_text_projection and name.startswith('text_projection'):
                    logger.info(f"Unfreeze {name}")
                elif self.config.model.open_text_lora and 'lora' in name:
                    logger.info(f"Unfreeze {name}")
                else:
                    logger.info(f"Freeze {name}")
                    p.requires_grad = False

        img_size = self.config.model.vision_encoder.img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        # load pretrained models
        self.load_checkpoint(
            config.model.vision_ckpt_path, config.model.text_ckpt_path, 
            config.model.get("extra_ckpt_path", None),
            config.model.get("finetuned_ckpt_path", None)
        )
        
        # criterions
        self.clip_loss = VTC_VTM_Loss(False)
        self.roi_loss = VTC_VTM_Loss(False)

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        # no weight decay for LLM if training
        ret.update(
            {"text_encoder." + k for k, _ in self.text_encoder.named_parameters()}
        )

        return ret
    
    def adjust_checkpoint(self, checkpoint):
        new_checkpoint = {}
        for key in checkpoint.keys():
            new_checkpoint[key] = checkpoint[key]
        
        for key in checkpoint.keys():
            if key.startswith('vision_encoder.'):
                key_parts = key.split('.')
                new_key = f"{key_parts[0]}.roi_branch.{'.'.join(key_parts[1:])}"
                new_checkpoint[new_key] = checkpoint[key]
        return new_checkpoint

    def adjust_checkpoint_f8(self, checkpoint):
        new_checkpoint = {}
        for key in checkpoint.keys():
            new_checkpoint[key] = checkpoint[key]
        
        for key in checkpoint.keys():
            if key.startswith('blocks.'):
                new_key = "roi_branch." + key
                new_checkpoint[new_key] = checkpoint[key]
        return new_checkpoint
    
    def adjust_checkpoint_f8_eval(self, checkpoint):
        new_checkpoint = {}
        for key in checkpoint.keys():
            new_checkpoint[key] = checkpoint[key]
        
        for key in checkpoint.keys():
            if key.startswith('blocks.'):
                new_key = "roi_branch." + key
                new_checkpoint[new_key] = checkpoint[key]
        return new_checkpoint
    
    @torch.no_grad()
    def clip_contrastive_temperature(self):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def forward(self, image, text, roi_box_list, labels=None, idx=None):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()
        vision_embeds, roi_embeds = self.encode_vision(image, roi_box_list)
        text_embeds = self.encode_text(text)

        # VTC loss
        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        loss_roi_vtc = self.roi_loss.vtc_loss(
            roi_embeds,  text_embeds, idx, self.temp, all_gather=True
        )

        loss_vtc = loss_vtc + loss_roi_vtc
        return dict(
            loss_vtc=loss_vtc,
        )

    def encode_vision(self, image, roi_box_list, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,C].

        """
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

        vision_embeds, roi_embeds = self.vision_encoder(image, roi_box_list, use_image=use_image)

        return vision_embeds, roi_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,C].

        """
        text_embeds = self.text_encoder(text)
        return text_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        vision_encoder = InternVideo2_1B_branch(
            in_chans=self.config.model.vision_encoder.in_chans,
            patch_size=self.config.model.vision_encoder.patch_size,
            img_size=self.config.model.vision_encoder.img_size,
            qkv_bias=self.config.model.vision_encoder.qkv_bias,
            drop_path_rate=self.config.model.vision_encoder.drop_path_rate,
            head_drop_path_rate=self.config.model.vision_encoder.head_drop_path_rate,
            embed_dim=self.config.model.vision_encoder.embed_dim,
            num_heads=self.config.model.vision_encoder.num_heads,
            mlp_ratio=self.config.model.vision_encoder.mlp_ratio,
            init_values=self.config.model.vision_encoder.init_values,
            qk_normalization=self.config.model.vision_encoder.qk_normalization,
            depth=self.config.model.vision_encoder.depth,
            use_flash_attn=self.config.model.vision_encoder.use_flash_attn,
            use_fused_rmsnorm=self.config.model.vision_encoder.use_fused_rmsnorm,
            use_fused_mlp=self.config.model.vision_encoder.use_fused_mlp,
            fused_mlp_heuristic=self.config.model.vision_encoder.fused_mlp_heuristic,
            attn_pool_num_heads=self.config.model.vision_encoder.attn_pool_num_heads,
            clip_embed_dim=self.config.model.vision_encoder.clip_embed_dim,
            layerscale_no_force_fp32=self.config.model.vision_encoder.layerscale_no_force_fp32,
            num_frames=self.config.model.vision_encoder.num_frames,
            tubelet_size=self.config.model.vision_encoder.tubelet_size,
            sep_pos_embed=self.config.model.vision_encoder.sep_pos_embed,
            use_checkpoint=self.config.model.vision_encoder.use_checkpoint,
            checkpoint_num=self.config.model.vision_encoder.checkpoint_num,
        )
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        text_encoder = LLaMA(
            use_flash_attn=self.config.model.text_encoder.use_flash_attn,
            transformer_width=self.config.model.text_encoder.transformer_width,
            llama_path=self.config.model.text_encoder.llama_path,
            use_lora=self.config.model.text_encoder.use_lora,
        )

        return text_encoder

    def load_checkpoint(self, vision_ckpt_path=None, text_ckpt_path=None, extra_ckpt_path=None, finetuned_ckpt_path=None):
        assert vision_ckpt_path is not None, "No vision_encoder checkpoint"
        assert text_ckpt_path is not None, "No text_encoder checkpoint"

        new_ckpt = {}

        # load vision_encoder
        logger.info(f"Load vision_encoder checkpoint from {vision_ckpt_path}")
        vision_ckpt = torch.load(vision_ckpt_path, map_location='cpu')
        if 'module' in vision_ckpt.keys():
            vision_ckpt = vision_ckpt['module']
        elif 'model' in vision_ckpt.keys():
            vision_ckpt = vision_ckpt['model']
        if not self.config.evaluate:
            vision_ckpt = self.adjust_checkpoint_f8(vision_ckpt)
        elif self.config.evaluate:
            vision_ckpt = self.adjust_checkpoint_f8_eval(vision_ckpt)
        if self.config.model.get('load_vision_ckpt_from_internvideo2_stage2', False):
            from .backbones.internvideo2.pos_embed import interpolate_pos_embed
            orig_t_size = self.config.model.get('vision_ckpt_t_size', 4)
            interpolate_pos_embed(vision_ckpt, self.vision_encoder, orig_t_size=orig_t_size) # 4 for InternVideo2 stage2
            for k, v in vision_ckpt.items():
                if k.startswith('vision_encoder.'):
                    if 'clip_decoder' in k or 'final_clip_decoder' in k:
                        continue
                    elif 'clip_pos_embed' in k or 'clip_img_pos_embed' in k or 'img_pos_embed' in k :
                        continue
                    else:
                        new_ckpt[k] = v
                else:
                    continue
        else:
            for k, v in vision_ckpt.items():
                if k.startswith('clip_decoder.') or k.startswith('mae_decoder.') or k.startswith('final_clip_decoder.'):
                    continue
                elif k in ['clip_pos_embed', 'mae_pos_embed']:
                    continue
                else:
                    new_k = 'vision_encoder.' + k
                    new_ckpt[new_k] = v

        # load text_encoder
        logger.info(f"Load text_encoder checkpoint from {text_ckpt_path}")
        text_ckpt = torch.load(text_ckpt_path, map_location='cpu')
        if 'module' in text_ckpt.keys():
            text_ckpt = text_ckpt['module']
        for k, v in text_ckpt.items():
            if k.startswith('transformer.') or k == 'text_projection':
                new_k = "text_encoder." + k
                if ("q_proj" in k or "v_proj" in k) and "lora" not in k:
                    # Correct handling of base_layer
                    new_k = new_k.replace(".q_proj", ".q_proj.base_layer").replace(".v_proj", ".v_proj.base_layer")
            else:
                continue
            new_ckpt[new_k] = v

        # load extra checkpoint
        # often when post-pretrain after previous pretraining, thus the keys are same
        if extra_ckpt_path is not None:
            logger.info(f"Load extra checkpoint from {extra_ckpt_path}")
            extra_ckpt = torch.load(extra_ckpt_path, map_location='cpu')
            if 'module' in extra_ckpt.keys():
                extra_ckpt = extra_ckpt['module']
            for k, v in extra_ckpt.items():
                new_ckpt[k] = v
        
        if finetuned_ckpt_path is not None:
            logger.info(f"Load finetuned checkpoint from {finetuned_ckpt_path}")
            finetuned_ckpt = torch.load(finetuned_ckpt_path)
            if 'module' in finetuned_ckpt.keys():
                finetuned_ckpt = finetuned_ckpt["module"]
            for k, v in finetuned_ckpt.items():
                new_ckpt[k] = v
        
        msg = self.load_state_dict(new_ckpt, strict=False)
        logger.info(msg)
