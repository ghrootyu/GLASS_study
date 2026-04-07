# =============================================================================
# GLASS 메인 모듈
#
# GLASS = GAS (Global Anomaly Synthesis) + LAS (Local Anomaly Synthesis)
#
# ── 전체 학습 흐름 ────────────────────────────────────────────────────────────
#
# [매 Epoch]
#   1. Center 계산:
#      모든 정상 이미지의 patch feature 평균 → 정상 feature 공간의 중심점 c
#
#   2. _train_discriminator (배치 단위 반복):
#
#      ┌── 입력 준비 ────────────────────────────────────────────────────────┐
#      │  img      : 정상 이미지 (3, 288, 288)                              │
#      │  aug      : Perlin+DTD 합성 이상 이미지 (3, 288, 288)  ← LAS      │
#      │  mask_s   : Perlin 마스크 (36, 36) pixel-level label   ← LAS      │
#      └─────────────────────────────────────────────────────────────────────┘
#
#      ┌── Feature 추출 (_embed) ────────────────────────────────────────────┐
#      │  img  → backbone → patchify → preprocessing → aggregator           │
#      │  → true_feats (B×1296, 1536)  정상 feature                         │
#      │                                                                     │
#      │  aug  → 동일 파이프라인                                              │
#      │  → fake_feats (B×1296, 1536)  합성 이상 feature        ← LAS      │
#      └─────────────────────────────────────────────────────────────────────┘
#
#      ┌── GAS: Gaussian Anomaly Synthesis ─────────────────────────────────┐
#      │  gaus_feats = true_feats + gaussian_noise                          │
#      │                                                                     │
#      │  Gradient Ascent (step=20회):                                       │
#      │    gaus_feats를 Discriminator 손실이 커지는 방향으로 이동           │
#      │    → Hypersphere shell [r, 2r] 위에 projection                     │
#      │    목적: 판별하기 어려운 경계 근처의 hard negative 생성             │
#      └─────────────────────────────────────────────────────────────────────┘
#
#      ┌── Loss 계산 ────────────────────────────────────────────────────────┐
#      │  BCE Loss  = BCE(true_feats→0) + BCE(gaus_feats→1)   ← GAS       │
#      │  Focal Loss = FocalLoss(fake_feats, mask_s)           ← LAS       │
#      │  Total Loss = BCE Loss + Focal Loss                                │
#      └─────────────────────────────────────────────────────────────────────┘
#
#   3. 매 eval_epochs마다 검증 데이터로 성능 평가 → best model 저장
# =============================================================================

from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class TBWrapper:
    """TensorBoard SummaryWriter 래퍼 (global iteration 카운터 포함)."""
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device

    def load(
            self,
            backbone,                        # 사전학습 backbone (WideResNet50)
            layers_to_extract_from,          # ["layer2", "layer3"]
            device,
            input_shape,                     # (3, 288, 288)
            pretrain_embed_dimension,        # 각 레이어 → 이 차원으로 압축 (1536)
            target_embed_dimension,          # 최종 feature 차원 (1536)
            patchsize=3,                     # patch 크기 (k×k), 3×3 이웃 포함
            patchstride=1,                   # patch sliding stride (1=모든 위치)
            meta_epochs=640,                 # 총 학습 epoch 수
            eval_epochs=1,                   # 몇 epoch마다 검증할지
            dsc_layers=2,                    # Discriminator 레이어 수
            dsc_hidden=1024,                 # Discriminator hidden 차원
            dsc_margin=0.5,                  # 정상/이상 판정 임계값 (모니터링용)
            train_backbone=False,            # backbone 파인튜닝 여부 (기본 False)
            pre_proj=1,                      # Projection 레이어 수 (1=사용, 0=미사용)
            mining=1,                        # Gradient Ascent hard mining 사용 여부
            noise=0.015,                     # GAS 초기 Gaussian 노이즈 표준편차
            radius=0.75,                     # Hypersphere radius quantile (75%)
            p=0.5,                           # Hard mining 비율 (상위 50%)
            lr=0.0001,                       # learning rate
            svd=0,                           # 0=Manifold, 1=Hypersphere 모드
            step=20,                         # Gradient Ascent 반복 횟수
            limit=392,                       # epoch당 최대 처리 샘플 수
            **kwargs,
    ):
        """
        모델 구성 요소 초기화.

        ── 모듈 구조 ──────────────────────────────────────────────────────────
        forward_modules:
            feature_aggregator: backbone hook으로 layer2, layer3 feature 추출
            preprocessing:      레이어별 feature를 1536 차원으로 압축
            preadapt_aggregator: 레이어 평균 → (B×1296, 1536) 최종 feature

        학습 대상 (gradient 흐름):
            pre_projection: feature 공간 변환 (Adam, lr=1e-4)
            discriminator:  patch 단위 이상 분류 (AdamW, lr=2e-4)
            backbone:       train_backbone=True일 때만 (AdamW, lr=1e-4)
        """

        self.backbone               = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape            = input_shape
        self.device                 = device

        # ── Feature 추출 파이프라인 ────────────────────────────────────────
        self.forward_modules = torch.nn.ModuleDict({})

        # backbone hook: layer2, layer3 feature 추출
        # WideResNet50: layer2=(B,512,36,36), layer3=(B,1024,18,18)
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        # 각 레이어의 채널 수 확인: [512, 1024]
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # 레이어별 feature → 동일 차원(1536)으로 압축
        # [512, 1024] → [1536, 1536]
        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension

        # 레이어 평균: (B×1296, 2, 1536) → (B×1296, 1536)
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # ── 학습 파라미터 ──────────────────────────────────────────────────
        self.meta_epochs    = meta_epochs
        self.lr             = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            # backbone 파인튜닝 시 별도 optimizer
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(), lr
            )

        # ── Pre-Projection (feature 공간 변환) ────────────────────────────
        # backbone feature를 이상 탐지에 적합한 공간으로 선형 변환
        # (B×1296, 1536) → (B×1296, 1536)
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension,
                self.target_embed_dimension,
                pre_proj,
            )
            self.pre_projection.to(self.device)
            # weight_decay로 과적합 방지
            self.proj_opt = torch.optim.Adam(
                self.pre_projection.parameters(), lr, weight_decay=1e-5
            )

        # ── Discriminator ─────────────────────────────────────────────────
        # patch feature → 이상 확률 [0, 1]
        # lr*2 = 2e-4 (Discriminator를 Projection보다 빠르게 학습)
        self.eval_epochs   = eval_epochs
        self.dsc_layers    = dsc_layers
        self.dsc_hidden    = dsc_hidden
        self.discriminator = Discriminator(
            self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden
        )
        self.discriminator.to(self.device)
        self.dsc_opt    = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        # ── GAS 관련 파라미터 ─────────────────────────────────────────────
        self.c        = torch.tensor(0)  # 정상 feature 중심점 (매 epoch 갱신)
        self.c_       = torch.tensor(0)  # 미사용 (예비)
        self.p        = p                # hard mining 비율 (상위 p=50%)
        self.radius   = radius           # hypersphere 반경 quantile (75%)
        self.mining   = mining           # gradient ascent 사용 여부
        self.noise    = noise            # GAS 초기 노이즈 크기 (0.015)
        self.svd      = svd              # 0=Manifold, 1=Hypersphere
        self.step     = step             # gradient ascent 반복 횟수 (20)
        self.limit    = limit            # epoch당 최대 샘플 수 (392)
        self.distribution = 0

        # LAS 손실
        self.focal_loss = FocalLoss()

        # ── Patch 처리 도구 ────────────────────────────────────────────────
        # patchsize=3: 각 feature 위치의 3×3 이웃을 하나의 patch로
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        # patch score (36×36) → 원본 크기 (288×288) upsample
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.model_dir    = ""
        self.dataset_name = ""
        self.logger       = None

    def set_model_dir(self, model_dir, dataset_name):
        """저장 경로 설정 및 TensorBoard 초기화."""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """
        이미지를 patch feature로 변환하는 핵심 파이프라인.

        ── 처리 흐름 ──────────────────────────────────────────────────────────
        입력: (B, 3, 288, 288)

        1. backbone feature 추출 (hook)
           layer2: (B, 512,  36, 36)
           layer3: (B, 1024, 18, 18)

        2. patchify (각 레이어 독립적으로)
           layer2: (B×1296, 512,  3, 3)  ← 36×36=1296 positions, 3×3 neighborhood
           layer3: (B×1296, 1024, 3, 3)  ← 18×18 → 36×36으로 보간 후 patchify

        3. layer3를 layer2 크기(36×36)로 bilinear 보간
           → 두 레이어의 spatial 크기를 통일

        4. Preprocessing (MeanMapper per layer)
           layer2: (B×1296, 512, 3, 3)  → adaptive_avg_pool → (B×1296, 1536)
           layer3: (B×1296, 1024, 3, 3) → adaptive_avg_pool → (B×1296, 1536)
           stack → (B×1296, 2, 1536)

        5. Aggregator (layer 평균)
           (B×1296, 2, 1536) → (B×1296, 1536)

        출력: patch_features (B×1296, 1536), patch_shapes [[36, 36], [18, 18]]

        Args:
            images:               (B, 3, 288, 288)
            evaluation:           True=추론 모드 (no_grad 내에서 호출됨)

        예시 (B=16):
            입력:  (16, 3, 288, 288)
            출력:  (20736, 1536)   20736 = 16 × 1296
        """
        # ── 1. backbone feature 추출 ──────────────────────────────────────
        # train_backbone=False(기본)이면 항상 eval + no_grad
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)
        # features = {"layer2": (B,512,36,36), "layer3": (B,1024,18,18)}

        # 지정된 레이어 순서대로 리스트화
        features = [features[layer] for layer in self.layers_to_extract_from]
        # [(B,512,36,36), (B,1024,18,18)]

        # ── ViT 계열 backbone 호환 처리 (sequence → spatial) ─────────────
        # CNN backbone(WideResNet50)은 이미 4D이므로 해당 없음
        # ViT는 (B, seq_len, C) 형태로 출력 → (B, C, H, W)로 변환
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(
                    B, int(math.sqrt(L)), int(math.sqrt(L)), C
                ).permute(0, 3, 1, 2)

        # ── 2. Patchify ───────────────────────────────────────────────────
        # 각 레이어 feature를 3×3 sliding window patch로 분해
        # layer2: (B,512,36,36)  → (B,1296,512,3,3),  patch_shapes=[36,36]
        # layer3: (B,1024,18,18) → (B,324,1024,3,3),  patch_shapes=[18,18]
        features    = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]   # [[36,36], [18,18]]
        patch_features = [x[0] for x in features] # patch tensors

        # layer2 patch 격자 크기를 기준으로 사용 (ref = [36, 36])
        ref_num_patches = patch_shapes[0]

        # ── 3. layer3 patch를 layer2 크기(36×36)로 보간 ──────────────────
        # layer2와 layer3의 spatial 크기가 다르므로 통일 필요
        # layer3 patch: (B,324,1024,3,3) → (B,1296,1024,3,3)
        for i in range(1, len(patch_features)):
            _features  = patch_features[i]   # (B, H2×W2, C, k, k) = (B,324,1024,3,3)
            patch_dims = patch_shapes[i]      # [18, 18]

            # reshape: (B, 18, 18, C, k, k)
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            # permute: (B, C, k, k, 18, 18) (공간 차원을 마지막으로)
            _features = _features.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = _features.shape

            # bilinear 보간을 위해 2D spatial 차원만 분리
            _features = _features.reshape(-1, *_features.shape[-2:])   # (B×C×k×k, 18, 18)
            _features = F.interpolate(
                _features.unsqueeze(1),                                  # (N, 1, 18, 18)
                size=(ref_num_patches[0], ref_num_patches[1]),           # (36, 36)
                mode="bilinear",
                align_corners=False,
            )                                                            # (N, 1, 36, 36)
            _features = _features.squeeze(1)                            # (N, 36, 36)

            # 원래 차원 구조로 복원
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, 4, 5, 1, 2, 3)            # (B, 36, 36, C, k, k)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            # (B, 1296, 1024, 3, 3)
            patch_features[i] = _features

        # ── 4. Preprocessing: 레이어별 feature → 1536 차원 압축 ──────────
        # [(B,1296,512,3,3), (B,1296,1024,3,3)] → 각 reshape → (B×1296, C, k, k)
        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        # [(20736, 512,3,3), (20736, 1024,3,3)]

        # MeanMapper: (N, C, k, k) → (N, 1536) per layer
        # stack: (N, 2, 1536)
        patch_features = self.forward_modules["preprocessing"](patch_features)

        # ── 5. Aggregator: 레이어 평균 ────────────────────────────────────
        # (N, 2, 1536) → (N, 1536)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes
        # patch_features: (B×1296, 1536)
        # patch_shapes:   [[36,36], [18,18]]

    def trainer(self, training_data, val_data, name):
        """
        GLASS 학습 메인 루프.

        ── 전체 흐름 ──────────────────────────────────────────────────────────
        1. 기존 ckpt 확인 → 있으면 학습 스킵
        2. distribution 모드 결정 (Manifold vs Hypersphere)
        3. distribution==1이면 FFT 분석으로 svd 자동 결정 후 조기 반환
        4. meta_epochs 반복:
            a. 정상 feature의 center(c) 계산
            b. _train_discriminator: Discriminator + Projection 학습
            c. eval_epochs마다 검증 → best_record + ckpt 갱신

        Args:
            training_data: train DataLoader (정상 이미지 + 합성 이상)
            val_data:      test DataLoader (검증용)
            name:          "mpdd_bracket_black" 형태의 데이터셋 이름

        반환:
            best_record: [i_auroc, i_ap, p_auroc, p_ap, p_pro, best_epoch]
            또는 int(svd): distribution==1일 때 svd 판정 결과
        """
        state_dict = {}

        # ── 기존 체크포인트 확인 ──────────────────────────────────────────
        # ckpt_best_*.pth가 있으면 이미 학습된 것으로 간주 → 학습 건너뜀
        ckpt_path      = glob.glob(self.ckpt_dir + '/ckpt_best*')
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            """현재 모델 가중치를 state_dict에 저장 (CPU로 이동)."""
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()
            })
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()
                })

        # ── distribution 모드 결정 ────────────────────────────────────────
        # svd=0: Manifold 모드 (복잡한 구조적 클래스에 적합)
        # svd=1: Hypersphere 모드 (균일한 텍스처 클래스에 적합)
        self.distribution = training_data.dataset.distribution
        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        try:
            if self.distribution == 1:
                # FFT 분석으로 자동 판별 (아래 별도 처리)
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:
                # 강제로 Manifold 모드
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:
                # 강제로 Hypersphere 모드
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:
                # excel 파일의 결과 반전 사용
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:
                # excel 파일에서 읽기 (이전 실행 결과 재사용)
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except:
            # excel 파일 없으면 FFT 자동 판별 모드로 fallback
            self.distribution = 1
            self.svd = 1

        # ── distribution==1: FFT 분석으로 svd 자동 결정 ──────────────────
        # 학습 이미지 평균을 구한 뒤 주파수 분석으로 텍스처 특성 판별
        # 주파수가 넓게 퍼진 클래스 → Hypersphere(svd=1)
        # 주파수가 중앙에 집중된 클래스 → Manifold(svd=0)
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    # 배치 평균을 누적하여 전체 평균 이미지 계산
                    batch_mean = torch.mean(img, dim=0)  # (3, 288, 288)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)  # (3, 288, 288) 전체 이미지 평균

            # 평균 이미지를 numpy로 변환 후 FFT 분석
            avg_img  = utils.torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = utils.distribution_judge(avg_img, name)  # 0 or 1

            # 판별 결과 이미지 저장 (시각적 확인용)
            os.makedirs(f'./results/judge/avg/{self.svd}', exist_ok=True)
            cv2.imwrite(f'./results/judge/avg/{self.svd}/{name}.png', avg_img)

            # svd 판정값(int)을 반환 → main에서 excel에 저장
            return self.svd

        # ── 학습 루프 ─────────────────────────────────────────────────────
        pbar       = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1  = ""   # 검증 성능 표시 문자열
        best_record = None

        for i_epoch in pbar:

            # ── Step 1: 정상 feature center(c) 계산 ──────────────────────
            # 매 epoch 시작 시 모든 정상 이미지의 patch feature 평균을 계산.
            # center c는 정상 분포의 중심점으로, GAS에서 hypersphere 기준점으로 사용.
            #
            # 계산 방법:
            #   각 배치의 patch feature 평균을 누적 → 전체 평균
            #
            # 예시 (B=16, 1296 patches):
            #   outputs shape: (16, 1296, 1536) → batch 평균 → (1296, 1536)
            #   모든 배치 누적 후 배치 수로 나눔
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)

                    # feature 추출 + projection
                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                        outputs = outputs[0] if len(outputs) == 2 else outputs
                    else:
                        outputs = self._embed(img, evaluation=False)[0]
                    outputs = outputs[0] if len(outputs) == 2 else outputs

                    # (B×1296, 1536) → (B, 1296, 1536) → batch 평균 → (1296, 1536)
                    outputs    = outputs.reshape(img.shape[0], -1, outputs.shape[-1])
                    batch_mean = torch.mean(outputs, dim=0)

                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean

                # 전체 배치 수로 나눠 평균 center 계산
                self.c /= len(training_data)
                # self.c shape: (1296, 1536) ← 1296 patch 위치 각각의 feature 평균

            # ── Step 2: Discriminator 학습 ────────────────────────────────
            pbar_str, pt, pf = self._train_discriminator(
                training_data, i_epoch, pbar, pbar_str1
            )
            update_state_dict()  # 현재 가중치 저장

            # ── Step 3: 검증 및 best model 저장 ──────────────────────────
            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(
                    images, scores, segmentations, labels_gt, masks_gt, name
                )

                # TensorBoard 기록
                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap",    image_ap,    i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap",    pixel_ap,    i_epoch)
                self.logger.logger.add_scalar("p-pro",   pixel_pro,   i_epoch)

                eval_path  = './results/eval/'     + name + '/'
                train_path = './results/training/' + name + '/'

                # best 판정 기준: image_auroc + pixel_auroc 합산
                if best_record is None:
                    # 첫 번째 eval → 무조건 best로 설정
                    best_record   = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)  # 시각화 결과 복사

                elif image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                    # 이전 best보다 좋으면 갱신
                    best_record    = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    os.remove(ckpt_path_best)              # 이전 best ckpt 삭제
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                # progress bar 표시 업데이트 (현재값 + 최고값)
                pbar_str1 = (
                    f" IAUC:{round(image_auroc*100,2)}({round(best_record[0]*100,2)})"
                    f" IAP:{round(image_ap*100,2)}({round(best_record[1]*100,2)})"
                    f" PAUC:{round(pixel_auroc*100,2)}({round(best_record[2]*100,2)})"
                    f" PAP:{round(pixel_ap*100,2)}({round(best_record[3]*100,2)})"
                    f" PRO:{round(pixel_pro*100,2)}({round(best_record[4]*100,2)})"
                    f" E:{i_epoch}({best_record[-1]})"
                )
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)

            # 매 epoch마다 최신 가중치 저장 (ckpt.pth = 마지막 epoch)
            torch.save(state_dict, ckpt_path_save)

        return best_record

    def _train_discriminator(self, input_data, cur_epoch, pbar, pbar_str1):
        """
        한 epoch의 Discriminator + Projection 학습.

        ── 배치 처리 흐름 ─────────────────────────────────────────────────────
        각 배치마다:
        1. img(정상), aug(합성이상), mask_s(Perlin마스크) 로드
        2. feature 추출: true_feats, fake_feats
        3. GAS: gaus_feats 생성 + Gradient Ascent hard mining
        4. BCE Loss (GAS): true_feats→0, gaus_feats→1
        5. Focal Loss (LAS): fake_feats → mask_s
        6. 역전파 + 파라미터 업데이트

        limit(=392) 초과 시 배치 루프 조기 종료 (epoch당 처리량 제한).

        Args:
            input_data: train DataLoader
            cur_epoch:  현재 epoch 번호
            pbar:       tqdm progress bar
            pbar_str1:  이전 eval 결과 문자열 (progress bar 표시용)

        반환: (pbar_str, mean_p_true, mean_p_fake)
            pbar_str:    현재 epoch 학습 통계 문자열
            mean_p_true: 정상 샘플 정확도 평균 (dsc_margin 기준)
            mean_p_fake: 이상 샘플 정확도 평균
        """
        # backbone, feature_aggregator는 고정 (eval 모드)
        self.forward_modules.eval()
        # 학습 대상만 train 모드
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_p_true, all_p_fake = [], [], []
        all_r_t, all_r_g, all_r_f = [], [], []
        sample_num = 0

        for i_iter, data_item in enumerate(input_data):
            # gradient 초기화
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            # ── 1. 데이터 로드 ────────────────────────────────────────────
            # aug:    (B, 3, 288, 288) Perlin+DTD 합성 이상 이미지  ← LAS
            # img:    (B, 3, 288, 288) 정상 이미지
            # mask_s: (B, 36, 36) Perlin 마스크 (1=이상 픽셀)       ← LAS
            aug    = data_item["aug"].to(torch.float).to(self.device)
            img    = data_item["image"].to(torch.float).to(self.device)

            # ── 2. Feature 추출 ───────────────────────────────────────────
            # _embed + pre_projection을 통해 patch feature 추출
            #
            # true_feats: (B×1296, 1536) 정상 이미지의 patch feature
            # fake_feats: (B×1296, 1536) 합성 이상 이미지의 patch feature  ← LAS
            #
            # 예시 (B=16): (20736, 1536)
            if self.pre_proj > 0:
                fake_feats = self.pre_projection(self._embed(aug, evaluation=False)[0])
                fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
            else:
                fake_feats = self._embed(aug, evaluation=False)[0]
                fake_feats.requires_grad = True
                true_feats = self._embed(img, evaluation=False)[0]
                true_feats.requires_grad = True

            # ── 3. mask_s reshape ─────────────────────────────────────────
            # mask_s: (B, 36, 36) → (B×1296, 1) patch-level label
            # 값: 0=정상 patch, 1=이상 patch (Perlin 마스크 기준)
            mask_s_gt = data_item["mask_s"].reshape(-1, 1).to(self.device)
            # 예시 (B=16): (20736, 1)

            # ── 4. GAS: Gaussian Noise로 초기 gaus_feats 생성 ────────────
            # true_feats에 작은 Gaussian 노이즈(std=0.015) 추가
            # → 정상 feature 공간 근처의 샘플 생성
            noise      = torch.normal(0, self.noise, true_feats.shape).to(self.device)
            gaus_feats = true_feats + noise
            # gaus_feats: (B×1296, 1536) 노이즈가 추가된 정상 feature

            # ── 5. Hypersphere 반경(r_t) 계산 ────────────────────────────
            # 정상 feature들이 center에서 얼마나 떨어져 있는지 측정
            # r_t: 75th percentile 거리 → 정상 분포의 "경계" 반경
            #
            # true_points: 정상 픽셀의 feature들
            #   = fake_feats 중 mask_s==0인 것(정상 픽셀) + 전체 true_feats
            # 이유: mask_s==0인 fake 픽셀은 합성 없이 원본과 동일 → 정상으로 취급
            center = self.c.repeat(img.shape[0], 1, 1)  # (B, 1296, 1536)
            center = center.reshape(-1, center.shape[-1]) # (B×1296, 1536)

            true_points = torch.concat([
                fake_feats[mask_s_gt[:, 0] == 0],  # 합성되지 않은 정상 픽셀
                true_feats,                          # 전체 정상 feature
            ], dim=0)
            c_t_points = torch.concat([
                center[mask_s_gt[:, 0] == 0],
                center,
            ], dim=0)

            # center까지의 거리 분포
            dist_t = torch.norm(true_points - c_t_points, dim=1)
            # r_t = 75th percentile 거리 (정상의 경계 반경)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)

            # ── 6. GAS: Gradient Ascent Hard Mining ──────────────────────
            # gaus_feats를 Discriminator가 이상으로 분류하기 어려운 위치로 이동.
            # 목적: 정상/이상 경계 근처의 어려운 샘플(hard negative) 생성
            #
            # 방법:
            #   gaus_loss(이상으로 분류 실패 시 커지는 손실)에 대한 gradient를 계산
            #   gradient 방향으로 gaus_feats를 조금씩 이동 (ascent)
            #   step=20회 반복 후 Hypersphere shell에 projection
            #
            # Hypersphere projection:
            #   svd=1: center 기준 [r, 2r] shell → 정상 바깥 경계에 배치
            #   svd=0: true_feats 기준 → Manifold 표면 근처에 배치
            for step in range(self.step + 1):
                # 현재 gaus_feats에 대한 Discriminator score 계산
                # true_feats와 gaus_feats를 함께 넣어 한 번에 처리
                scores      = self.discriminator(torch.cat([true_feats, gaus_feats]))
                true_scores = scores[:len(true_feats)]    # 정상 feature score
                gaus_scores = scores[len(true_feats):]    # gaus feature score

                # BCE Loss 계산 (GAS 손실)
                # true_feats → 0 (정상으로 분류되어야 함)
                # gaus_feats → 1 (이상으로 분류되어야 함)
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss  = true_loss + gaus_loss

                # 마지막 step이면 gradient ascent 없이 loss만 계산
                if step == self.step:
                    break
                elif self.mining == 0:
                    # mining=0: gradient ascent 미사용, 단순 Gaussian만 사용
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g    = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    break

                # ── Gradient Ascent: gaus_feats 이동 ─────────────────────
                # gaus_loss에 대한 gaus_feats의 gradient 계산
                # gradient 방향 = Discriminator가 더 이상으로 판정하는 방향
                grad      = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
                grad_norm = torch.norm(grad, dim=1).view(-1, 1)
                # gradient를 단위 벡터로 정규화 (방향만 사용)
                grad_normalized = grad / (grad_norm + 1e-10)

                # gradient 방향으로 0.001씩 이동
                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized)

                # ── 5 step마다 Hypersphere Projection ────────────────────
                # gaus_feats가 너무 멀리 가지 않도록 shell [r, 2r]에 제한
                if (step + 1) % 5 == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g    = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)

                    # svd=1 (Hypersphere): center 기준 projection
                    # svd=0 (Manifold):   true_feats 기준 projection
                    proj_feats = center     if self.svd == 1 else true_feats
                    r          = r_t        if self.svd == 1 else 0.5

                    # gaus_feats - proj_feats = 방향 벡터 h
                    h      = gaus_feats - proj_feats
                    h_norm = dist_g     if self.svd == 1 else torch.norm(h, dim=1)

                    # |h|를 [r, 2r] 범위로 클리핑 → shell 위에 projection
                    # alpha: 목표 거리 (r ≤ alpha ≤ 2r)
                    alpha = torch.clamp(h_norm, r, 2 * r)
                    proj  = (alpha / (h_norm + 1e-10)).view(-1, 1)
                    h     = proj * h
                    # 새 위치: proj_feats + 방향 × 목표거리
                    gaus_feats = proj_feats + h

            # ── 7. LAS: fake_feats projection ────────────────────────────
            # mask_s==1인 픽셀의 fake_feats(합성 이상)를 center 바깥으로 projection
            # 목적: 합성 이상이 정상 분포와 명확히 분리되도록 강제
            #
            # svd=1이면: center 기준 [2r, 4r] shell에 projection
            #           → 정상 hypersphere보다 훨씬 바깥에 위치
            # svd=0이면: projection 미적용
            fake_points = fake_feats[mask_s_gt[:, 0] == 1]   # 이상 픽셀 feature
            true_points = true_feats[mask_s_gt[:, 0] == 1]   # 동일 위치 정상 feature
            c_f_points  = center[mask_s_gt[:, 0] == 1]        # 동일 위치 center

            dist_f = torch.norm(fake_points - c_f_points, dim=1)
            r_f    = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)

            proj_feats = c_f_points if self.svd == 1 else true_points
            r          = r_t        if self.svd == 1 else 1

            if self.svd == 1:
                # fake_points를 [2r, 4r] shell로 projection
                # (GAS의 [r, 2r]보다 더 바깥 → 더 명확한 이상 위치)
                h      = fake_points - proj_feats
                h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
                alpha  = torch.clamp(h_norm, 2 * r, 4 * r)
                proj   = (alpha / (h_norm + 1e-10)).view(-1, 1)
                h      = proj * h
                fake_points = proj_feats + h
                # projection된 fake_points를 fake_feats에 반영
                fake_feats[mask_s_gt[:, 0] == 1] = fake_points

            # ── 8. Focal Loss 계산 (LAS) ──────────────────────────────────
            # fake_feats 전체에 대해 mask_s_gt를 label로 Focal Loss 계산
            # mask_s_gt: 0=정상 픽셀, 1=합성 이상 픽셀
            #
            # Hard mining (p=0.5):
            #   오차가 큰 상위 50% 샘플만 선택하여 학습
            #   이유: 쉬운 샘플(이미 잘 분류됨)은 loss 기여 제한
            fake_scores = self.discriminator(fake_feats)  # (B×1296, 1)

            if self.p > 0:
                # 오차 = (예측 - 정답)^2
                fake_dist = (fake_scores - mask_s_gt) ** 2
                # 상위 p% 오차 임계값
                d_hard = torch.quantile(fake_dist, q=self.p)
                # 어려운 샘플만 선택 (오차 ≥ d_hard)
                fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
                mask_        = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
            else:
                # hard mining 미사용 (전체 사용)
                fake_scores_ = fake_scores
                mask_        = mask_s_gt

            # Focal Loss 입력: (N, 2) = [정상확률(1-score), 이상확률(score)]
            output     = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
            focal_loss = self.focal_loss(output, mask_)

            # ── 9. 총 Loss = BCE(GAS) + Focal(LAS) → 역전파 ─────────────
            loss = bce_loss + focal_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()          # Projection 업데이트
            if self.train_backbone:
                self.backbone_opt.step()      # Backbone 업데이트 (선택적)
            self.dsc_opt.step()               # Discriminator 업데이트

            # ── 10. 모니터링 지표 계산 ────────────────────────────────────
            # p_true: 정상 샘플 중 score < dsc_margin(0.5)인 비율 (올바르게 정상 판정)
            # p_fake: 이상 샘플 중 score ≥ dsc_margin(0.5)인 비율 (올바르게 이상 판정)
            pix_true = torch.concat([
                fake_scores.detach() * (1 - mask_s_gt),  # 정상 픽셀의 score (mask_s==0)
                true_scores.detach(),                      # 전체 정상 feature score
            ])
            pix_fake = torch.concat([
                fake_scores.detach() * mask_s_gt,          # 이상 픽셀의 score (mask_s==1)
                gaus_scores.detach(),                       # GAS fake score
            ])
            p_true = (
                (pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()
            ) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / (
                (mask_s_gt == 1).sum() + gaus_scores.shape[0]
            )

            # TensorBoard 기록
            self.logger.logger.add_scalar("p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar("p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar("r_t",    r_t,    self.logger.g_iter)
            self.logger.logger.add_scalar("r_g",    r_g,    self.logger.g_iter)
            self.logger.logger.add_scalar("r_f",    r_f,    self.logger.g_iter)
            self.logger.logger.add_scalar("loss",   loss,   self.logger.g_iter)
            self.logger.step()

            # 통계 누적
            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_   = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_    = np.mean(all_r_t)
            all_r_g_    = np.mean(all_r_g)
            all_r_f_    = np.mean(all_r_f)
            sample_num  = sample_num + img.shape[0]

            # progress bar 업데이트
            # pt=p_true (정상 정확도%), pf=p_fake (이상 정확도%)
            # rt=정상 반경, rg=GAS 반경, rf=LAS 반경
            pbar_str  = f"epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_*100:.2f}"
            pbar_str += f" pf:{all_p_fake_*100:.2f}"
            pbar_str += f" rt:{all_r_t_:.2f}"
            pbar_str += f" rg:{all_r_g_:.2f}"
            pbar_str += f" rf:{all_r_f_:.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2  = pbar_str
            pbar_str  += pbar_str1
            pbar.set_description_str(pbar_str)

            # limit 초과 시 조기 종료 (epoch당 처리량 제한)
            # limit=392이면 약 24배치(16×24=384) 처리 후 종료
            if sample_num > self.limit:
                break

        return pbar_str2, all_p_true_, all_p_fake_

    def tester(self, test_data, name):
        """
        저장된 best checkpoint를 로드하여 최종 성능 평가.

        best ckpt 파일(ckpt_best_{epoch}.pth) 로드 후
        전체 test 데이터에 대해 추론 및 평가 지표 계산.

        반환: (image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, best_epoch)
        """
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(
                images, scores, segmentations, labels_gt, masks_gt, name, path='eval'
            )
            # 파일명에서 best epoch 번호 추출
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            image_auroc = image_ap = pixel_auroc = pixel_ap = pixel_pro = 0.
            epoch = -1
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        """
        추론 결과에 대한 성능 지표 계산 및 시각화 저장.

        ── 지표 계산 ───────────────────────────────────────────────────────────
        image_auroc: 이미지 단위 AUROC (항상 계산)
        image_ap:    이미지 단위 Average Precision (path='eval'일 때만)
        pixel_auroc: 픽셀 단위 AUROC (masks_gt가 있을 때)
        pixel_ap:    픽셀 단위 Average Precision (path='eval'일 때만)
        pixel_pro:   Per-Region Overlap AUROC (path='eval'일 때만, 비용이 큰 지표)

        ── 시각화 ──────────────────────────────────────────────────────────────
        각 test 이미지에 대해 3분할 이미지 저장:
        [원본 | GT 마스크 | 예측 히트맵(JET colormap)]
        → results/training/{name}/ (학습 중) 또는 results/eval/{name}/ (최종)

        Args:
            images:        list of (3, 288, 288) numpy  ← 원본 이미지
            scores:        list of scalar               ← 이미지별 이상 score
            segmentations: list of (288, 288) numpy     ← 픽셀별 이상 score
            labels_gt:     list of 0/1                  ← 이미지 레이블
            masks_gt:      list of (1, 288, 288) numpy  ← GT 픽셀 마스크
            path:          'training' or 'eval'
        """
        scores       = np.squeeze(np.array(scores))   # (N,)
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc  = image_scores["auroc"]
        image_ap     = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)   # (N, 288, 288)
            pixel_scores  = metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt, path
            )
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap    = pixel_scores["ap"]

            # PRO는 비용이 커서 최종 eval 시에만 계산
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(
                        np.squeeze(np.array(masks_gt)), segmentations
                    )
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = pixel_ap = pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        # ── 시각화 저장 ────────────────────────────────────────────────────
        defects = np.array(images)   # (N, 3, 288, 288)
        targets = np.array(masks_gt) # (N, 1, 288, 288)

        for i in range(len(defects)):
            # 정규화 역변환: (3, 288, 288) tensor → (288, 288, 3) BGR numpy
            defect = utils.torch_format_2_numpy_img(defects[i])
            target = utils.torch_format_2_numpy_img(targets[i])

            # segmentation → JET colormap 히트맵
            # 높은 score = 빨강(이상), 낮은 score = 파랑(정상)
            mask = cv2.cvtColor(
                cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0])),
                cv2.COLOR_GRAY2BGR,
            )
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            # 3분할 가로 연결 후 256×768로 리사이즈
            img_up = np.hstack([defect, target, mask])
            img_up = cv2.resize(img_up, (256 * 3, 256))

            full_path = './results/' + path + '/' + name + '/'
            utils.del_remake_dir(full_path, del_flag=False)
            # 파일명: 001.png, 002.png, ...
            cv2.imwrite(full_path + str(i + 1).zfill(3) + '.png', img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader):
        """
        전체 test DataLoader에 대해 추론 수행.

        각 배치를 _predict에 넘겨 이미지별 score와 segmentation 수집.

        반환:
            images:        list of (3, 288, 288)  원본 이미지
            scores:        list of scalar          이미지별 이상 score
            masks:         list of (288, 288)      픽셀별 이상 score (segmentation)
            labels_gt:     list of 0/1             GT 이미지 레이블
            masks_gt:      list of (1, 288, 288)   GT 픽셀 마스크
        """
        self.forward_modules.eval()
        img_paths  = []
        images     = []
        scores     = []
        masks      = []
        labels_gt  = []
        masks_gt   = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt

    def _predict(self, img):
        """
        이미지 배치에 대한 추론.

        ── 처리 흐름 ──────────────────────────────────────────────────────────
        입력: (B, 3, 288, 288)

        1. _embed: backbone → patch features (B×1296, 1536)
        2. pre_projection: feature 변환 (B×1296, 1536)
        3. discriminator: patch별 이상 score (B×1296, 1)
        4. unpatch + reshape: (B, 36, 36) patch score 격자
        5. convert_to_segmentation: bilinear upsample + gaussian → (B, 288, 288)
        6. image_score: patch score 중 최댓값 → 이미지별 scalar

        반환:
            image_scores: list of scalar (B개), 이미지별 이상 score
            masks:        list of (288, 288) (B개), 픽셀별 이상 score

        예시 (B=16):
            patch_features: (20736, 1536)
            patch_scores:   (20736, 1) → reshape → (16, 36, 36)
            masks:          list of 16개 (288, 288) segmentation map
            image_scores:   list of 16개 scalar
        """
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # feature 추출
            patch_features, patch_shapes = self._embed(
                img, provide_patch_shapes=True, evaluation=True
            )
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            # Discriminator: (B×1296, 1536) → (B×1296, 1)
            patch_scores = image_scores = self.discriminator(patch_features)

            # unpatch: (B×1296, 1) → (B, 1296, 1)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales       = patch_shapes[0]    # [36, 36]
            # reshape: (B, 1296, 1) → (B, 36, 36)
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])

            # segmentation: (B, 36, 36) → bilinear upsample → gaussian → list of (288, 288)
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            # image_score: (B×1296, 1) → (B, 1296, 1) → max → (B,)
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)  # max over patches
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
