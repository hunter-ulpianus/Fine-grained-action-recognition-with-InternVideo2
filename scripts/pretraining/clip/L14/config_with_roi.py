
from configs.data import *
from configs.model import *

# ========================= data ==========================
# train_corpus = "ssv2_imgs"
train_corups = "ssv1_imgs"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
# test_file = dict(mc_test=available_corpus["ssv2_img_val"])
test_file = dict(mc_test=available_corpus["ssv1_img_val"])
test_types = ["ret_val"]
num_workers = 1

stop_key = None

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 4
batch_size_test = 32
max_txt_l = 32

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video_img="${batch_size}"),
    # batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video_img="${batch_size_test}"),
    # batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
model = dict(
    model_cls="InternVideo2_CLIP_small",
    vision_encoder=dict(
        name="internvideo2",
        in_chans=3,
        patch_size=14,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.,
        head_drop_path_rate=0.,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4,
        init_values=0.1,
        qk_normalization=True,
        depth=24,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        drop_cls_token=False,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        checkpoint_num=0,
        align_dim=512,
    ),
    text_encoder=dict(
        name="mobileclip_b"
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_vision=True,
    open_vision_clip_projector=True,
    freeze_text=True,
    train_branch=False,
    finetuned_pth = "",
    partial_freeze=False,
    partial_freezed_num=20,
    open_text_projection=False,
    open_text_lora=False,
    vision_ckpt_path= "/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/ckpts/mobileclip_vision.bin",
    load_vision_ckpt_from_internvideo2_stage2=False,
    # vision_ckpt_path = "/media/sdc/fe/InternVideo/InternVideo2/multi_modality/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt",
    text_ckpt_path= "/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/ckpts/mobileclip_blt.pt",
    extra_ckpt_path = "/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/ckpts/mobileclip_extra.bin"
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
    ),  # 0: disabled.
)

optimizer = dict(
    opt="adamW",
    lr=4e-4,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.2,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=True, module_names=["roi_attn"], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.6)

evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="754487925@qq.com",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="InternVideo2_CLIP",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = "/media/sdc/fe/ssv1_L14"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
#save_iter = 500
auto_resume = False
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=False,
    stage=0,
)