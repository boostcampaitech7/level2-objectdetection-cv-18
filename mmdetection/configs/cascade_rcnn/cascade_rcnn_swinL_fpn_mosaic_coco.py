_base_ = './cascade_rcnn_r50_fpn_20e_coco.py'
load_from = '/data/ephemeral/home/euna/level2-objectdetection-cv-18/mmdetection/work_dirs/cascade_rcnn_mosaic_trash/_mAP50_0.2970_epoch_11_.pth'
pretrained = None

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),  # 네 개의 stage 출력
        with_cp=False,
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 192*2, 192*4, 192*8],  # Swin-L 백본에서 나오는 출력 채널 수
        out_channels=256,
        num_outs=5
    )
)

# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    # Mosaic 증강을 추가
    dict(
        type='Mosaic',
        img_scale=(1024, 1024),  # Mosaic 적용 후 리사이즈할 크기
        center_ratio_range=(0.5, 1.5),  # 각 이미지 중심의 비율 범위 설정
        bbox_clip_border=True, # 이미지 경계 밖 bbox 잘라내기
        pad_val=114,  # 패딩 값 설정 (일반적으로 중성 회색 값)
        prob=0.5  # 50% 확률로 Mosaic 적용 
    ),

    dict(type='Resize', img_scale=[(1024, 720), (1024, 1024)],
        multiscale_mode='value',backend='pillow'),

    # dict(
    #     type='RandomCrop',
    #     crop_size=(720, 720),  # 크롭할 영역의 크기 설정
    #     crop_type='absolute',  # 절대 픽셀 크기로 크롭
    #     allow_negative_crop=False,  # bbox가 없는 크롭 허용 안 함
    #     bbox_clip_border=True  # 이미지 경계 밖 bbox 잘라내기
    # ),

    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1024, 720), (1024, 1024)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data = dict(
#     train=dict(
#         type='MultiImageMixDataset',
#         dataset=dict(  # 실제 데이터셋 정보는 이 안에 정의
#             type='CocoDataset',  # CocoDataset 형식
#             ann_file='/data/ephemeral/home/dataset/train_split_0.json',  # 어노테이션 파일 경로
#             img_prefix='/data/ephemeral/home/dataset/',  # 이미지 경로
#             pipeline=[  # 원본 데이터셋에 대한 파이프라인 (이미지 로딩 및 어노테이션 로딩)
#                 dict(type='LoadImageFromFile'),
#                 dict(type='LoadAnnotations', with_bbox=True),
#             ]
#         ),
#         pipeline=train_pipeline  # Mosaic과 같은 추가적인 파이프라인 적용
#     ),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline)
# )