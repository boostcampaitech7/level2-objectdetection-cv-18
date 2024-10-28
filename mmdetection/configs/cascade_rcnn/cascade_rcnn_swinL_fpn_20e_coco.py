_base_ = './cascade_rcnn_r50_fpn_20e_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),  # 네 개의 stage 출력
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],  # Swin-L 백본에서 나오는 출력 채널 수
        out_channels=256,
        num_outs=5
    )
)
