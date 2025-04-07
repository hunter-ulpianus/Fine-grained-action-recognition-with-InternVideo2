from configs.data import *
from configs.model import *

# ========================= data ==========================
train_corpus = "ssv2_imgs"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict(mc_test=available_corpus["ssv2_img_val"])
test_types = ["mc_test"]
num_workers = 12

stop_key = None

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 1
batch_size_test = 2
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
    batch_size_test=dict(image="${batch_size_test}", video_img="${batch_size_test}"),
)

# ========================= model ==========================
model = dict(
    # model_cls="InternVideo2_CLIP",
    model_cls="InternVideo2_1Broi_branch",
    vision_encoder=dict(
        name="internvideo2",
        in_chans=3,
        patch_size=14,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.3,
        head_drop_path_rate=0.,
        embed_dim=1408,
        num_heads=16,
        mlp_ratio=48/11,
        init_values=0.1,
        qk_normalization=True,
        depth=40,
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
    ),
    text_encoder=dict(
        use_flash_attn=True,
        transformer_width=4096,
        llama_path="/media/sdc/fe/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b/",
        use_lora=True,
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_vision=True,
    open_vision_clip_projector=True,   # default True
    freeze_text=True,
    open_text_projection=False,
    open_text_lora=False,
    tokenizer_path="/media/sdc/fe/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b/",
    # vision_ckpt_path="/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/ckpts/InternVideo2-stage2_1b-224p-f4.pt",
    vision_ckpt_path="/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/ckpts/1B_ft_ssv2_f8.pth",
    # vision_ckpt_path = "/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/ckpts/1B_ft_ssv2_f8.pth",
    load_vision_ckpt_from_internvideo2_stage2=False,
    text_ckpt_path="/media/sdc/fe/InternVideo/InternVideo2/multi_modality/internvl_c_13b_224px.pth",
    extra_ckpt_path = "/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/ckpts/1B_clip.pth",
    finetuned_ckpt_path = "/media/sdc/fe/ckpt_1B_250303/ckpt_05.pth/mp_rank_00_model_states.pt"
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
    ),  # 0: disabled.
)

optimizer = dict(
    opt="adamW",
    # lr=4e-4,
    lr=4e-4,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.2,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=True, module_names=[""], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=10, min_lr_multi=0.01, warmup_epochs=0.6)

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
output_dir = "/media/sdc/fe/ckpt_1B_250321_2stage/"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
# save_iter = 500
auto_resume = False
pretrained_path = ""
stage2_train = True
# ========================= deepspeed ==========================
deepspeed = dict(
    enable=True,
    stage=1,
    zero_optimization=dict(
        stage=0,
        offload_optimizer=False,
        offload_parameters=False,
        nvme_offload_params=False,
        nvme_offload_optimizer=False,
    ),
)
