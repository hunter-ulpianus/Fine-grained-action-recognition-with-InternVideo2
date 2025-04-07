import logging
import os
import json

import torch
from torch import nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from .backbones.internvideo2 import InternVideo2, TextTransformer, ClipTokenizer
# from .backbones.internvideo2 import InternVideo2_roi, TextTransformer, ClipTokenizer
from .criterions import VTC_VTM_Loss


logger = logging.getLogger(__name__)


class InternVideo2_CLIP_small(nn.Module):
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain
        self.vision_width = self.config.model.vision_encoder.clip_embed_dim
        self.embed_dim = 1024

        # create modules.
        """
        text_encoder_cfg = json.load(
            open(os.path.join(
                "./models/backbones/internvideo2/mobileclip/configs/" + \
                f"{self.config.model.text_encoder.name}.json"))
        )
        """

        text_encoder_cfg = json.load(
            open("/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/models/backbones/internvideo2/mobileclip/configs/mobileclip_b.json")
        )
        if tokenizer is None:
            self.tokenizer = ClipTokenizer(text_encoder_cfg)
        # self.segmentor = self.build_sengmentor()
        self.vision_encoder = self.build_vision_encoder()
        self.vision_align = nn.Sequential(
            nn.LayerNorm(self.config.model.vision_encoder.clip_embed_dim),
            nn.Linear(
                self.config.model.vision_encoder.clip_embed_dim, 
                self.config.model.vision_encoder.align_dim
            ),
        )
        self.text_encoder = self.build_text_encoder(cfg=text_encoder_cfg['text_cfg'], projection_dim=text_encoder_cfg["embed_dim"])
        # adopt 1 / 100. as in ViCLIP
        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.temp_min = config.model.temp_min
        
        if self.config.model.partial_freeze and self.config.model.freeze_vision:
            for name, p in self.vision_encoder.named_parameters():
                if "layer" in name and int(name.split("."[1]) < self.config.model.partial_freezed_num):
                    p.requires_grad = False
            print(f"Top {self.config.model.partial_freezed_num} layers are freezed!")

        # freeze model
        if self.config.model.freeze_vision:
            for name, p in self.vision_encoder.named_parameters():
                if self.config.model.open_vision_clip_projector and name.startswith('clip_projector'):
                    logger.info(f"Unfreeze {name}")
                else:
                    logger.info(f"Freeze {name}")
                    p.requires_grad = False
                
                if self.config.model.train_branch:
                    if "roi" in name or "global_attn" in name or "alpha" in name or "multi_scale" in name or "adapter" in name or "fusion" in name or "high" in name:
                        p.requires_grad = True
                        logger.info(f"Train new-added module {name}")
                        test = None

        if self.config.model.freeze_text:
            for name, p in self.text_encoder.named_parameters():
                if self.config.model.open_text_projection and name.startswith('projection_layer'):
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
            config.model.get("extra_ckpt_path", None)
        )
        
        # criterions
        self.clip_loss = VTC_VTM_Loss(False)

        # recognition
        self.num_classes = 174
    """
        self.action_classifier = nn.Sequential(
            nn.Linear(self.config.model.vision_encoder.align_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )
        """

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
    
    @torch.no_grad()
    def clip_contrastive_temperature(self):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)
    
    def predict_label_loss(self, 
                      vid_feat: torch.Tensor, 
                      txt_feat: torch.Tensor, 
                      top: int=5):
        label_probs = (100.0 * vid_feat @ txt_feat.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.float().cpu().topk(top, dim=-1)
        return top_probs, top_labels
    
    
    # def forward(self, image, text, roi_box_list, labels=None, idx=None):
    def forward(self, image, text, labels=None, idx=None): 
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()
        # vision_embeds = self.encode_vision(image, roi_box_list)
        vision_embeds = self.encode_vision(image)
        text_embeds = self.encode_text(text)

        # VTC loss
        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        """
        vision_embeds_agg = vision_embeds
        class_logits = self.action_classifier(vision_embeds_agg)

        loss_cls = 0

        if labels is not None:
            action_criterion = nn.CrossEntropyLoss()
            loss_cls = action_criterion(class_logits, labels)
        
        loss_final = 0.5 * loss_cls + loss_vtc
        """
        return dict(
            loss_vtc=loss_vtc,
        )

    def roi_encode(self, image, obj_to_seg):
        return
    

    # def encode_vision(self, image, roi_box_list, test=False):
    def encode_vision(self, image, test=False):
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

        # roi_patches = self.roi_encode(image, obj_to_seg)
        # vision_embeds = self.vision_encoder(image, roi_box_list, use_image=use_image)
        vision_embeds = self.vision_encoder(image, use_image=use_image)
        vision_embeds = self.vision_align(vision_embeds)
        return vision_embeds

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
    """
    def build_segmentor(self):
        segmentor = GdSAM2()
        return
        """

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        vision_encoder = InternVideo2(
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

    def build_text_encoder(self, cfg, projection_dim):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        text_encoder = TextTransformer(cfg, projection_dim)

        return text_encoder

    def adjust_checkpoint_val(self, checkpoint):
        new_checkpoint = {}
        for key in checkpoint.keys():
            if key.startswith('blocks.') and not 'adpter' in key:
                block_idx = key.split('.')[1]
                if block_idx.isdigit() and int(block_idx) > 13:
                    new_key = f'blocks.{block_idx}' + '.block' + '.' + '.'.join(key.split('.')[2:])
                    new_checkpoint[new_key] = checkpoint[key]
                else:
                    new_checkpoint[key] = checkpoint[key]
            elif "align" in key:
                new_key = "vision_encoder." + key
                new_checkpoint[new_key] = checkpoint[key]
            else:
                new_checkpoint[key] = checkpoint[key]
        new_checkpoint['multi_scale_patch_embed.patch_embed_high.proj.weight'] = checkpoint['patch_embed.proj.weight']
        new_checkpoint['multi_scale_patch_embed.patch_embed_high.proj.bias'] = checkpoint['patch_embed.proj.bias']

        return new_checkpoint

    def adjust_checkpoint(self, checkpoint):
        new_checkpoint = {}
        for key in checkpoint.keys():
            if key.startswith('blocks.') and not 'adpter' in key:
                block_idx = key.split('.')[1]
                if block_idx.isdigit() and int(block_idx) > 13:
                    new_key = f'blocks.{block_idx}' + '.block' + '.' + '.'.join(key.split('.')[2:])
                    new_checkpoint[new_key] = checkpoint[key]
                else:
                    new_checkpoint[key] = checkpoint[key]
            else:
                new_checkpoint[key] = checkpoint[key]
        new_checkpoint['multi_scale_patch_embed.patch_embed_high.proj.weight'] = checkpoint['patch_embed.proj.weight']
        new_checkpoint['multi_scale_patch_embed.patch_embed_high.proj.bias'] = checkpoint['patch_embed.proj.bias']

        return new_checkpoint
    
    def load_checkpoint(self, vision_ckpt_path=None, text_ckpt_path=None, extra_ckpt_path=None):
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
        
        if extra_ckpt_path is not None :
            if 'ckpt' not in extra_ckpt_path:
                logger.info(f"Load extra checkpoint from {extra_ckpt_path}")
                extra_ckpt = torch.load(extra_ckpt_path, map_location='cpu')
                if 'module' in extra_ckpt.keys():
                    extra_ckpt = extra_ckpt['module']
                for k, v in extra_ckpt.items():
                    new_ckpt[k] = v
            else:
                logger.info(f"Load fine-tuned extra checkpoint from {extra_ckpt_path}")
                extra_ckpt = torch.load(extra_ckpt_path, map_location='cpu')
                if 'module' in extra_ckpt.keys():
                    extra_ckpt = extra_ckpt['module']
                elif 'model' in extra_ckpt.keys():
                    extra_ckpt = extra_ckpt['model']
                for k, v in extra_ckpt.items():
                    new_ckpt[k] = v
                
        if self.config.evaluate:
            # vision_ckpt = self.adjust_checkpoint_val(vision_ckpt)
            logger.info(f"load finetuned ckpt from {self.config.model.finetuned_pth}")
            finetuned_state = torch.load(self.config.model.finetuned_pth, map_location='cpu')
            if 'module' in finetuned_state.keys():
                for key in finetuned_state["module"].keys():
                    vision_ckpt[key] = finetuned_state["module"][key]
            else:
                for key in finetuned_state.keys():
                    vision_ckpt[key] = finetuned_state[key]
        """
        else:
            vision_ckpt = self.adjust_checkpoint(vision_ckpt)
            """
            
            
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
                elif self.config.evaluate:
                    if 'vision_encoder' in k:
                        new_ckpt[k] = v
                    elif "vision_align" in k or "action_classifier" in k or "fc_norm" in k or "temp" in k or "head" in k:
                        new_ckpt[k] = v
                    else:
                        new_k = "vision_encoder." + k
                        new_ckpt[new_k] = v
                else:
                    new_k = 'vision_encoder.' + k
                    new_ckpt[new_k] = v

        # load text_encoder
        logger.info(f"Load text_encoder checkpoint from {text_ckpt_path}")
        test_ckpt = torch.load(text_ckpt_path, map_location='cpu')
        if 'module' in test_ckpt.keys():
            test_ckpt = test_ckpt['module']
        for k, v in test_ckpt.items():
            if k.startswith('text_encoder.'):
                new_ckpt[k] = v

        # load extra checkpoint
        # often when post-pretrain after previous pretraining, thus the keys are same
        
        msg = self.load_state_dict(new_ckpt, strict=False)
        # msg = self.load_state_dict(vision_ckpt)
        logger.info(msg)
        test = None
