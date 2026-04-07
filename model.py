# =============================================================================
# GLASS 모델 구성 요소
#
# Discriminator: patch feature를 받아 정상(0)/이상(1) 확률 출력
# Projection:    feature를 같은 차원으로 선형 변환 (pre_projection)
# PatchMaker:    feature map을 patch 단위로 분해/복원
# =============================================================================

import torch


def init_weight(m):
    """가중치 초기화 (Xavier normal for Linear, Normal for BN/Conv)"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


class Discriminator(torch.nn.Module):
    """
    Patch-level 이진 분류기.

    각 patch feature를 입력받아 정상(0) / 이상(1) 확률을 출력.
    GLASS의 핵심 학습 대상 모듈.

    구조 (dsc_layers=2, dsc_hidden=1024, in_planes=1536):
        Linear(1536 → 1024) + BN + LeakyReLU   ← body block 1
        Linear(1024 → 1, bias=False) + Sigmoid  ← tail

    학습 대상:
        - true_feats (정상 feature)  → 출력 0
        - gaus_feats (GAS fake)      → 출력 1
        - fake_feats (LAS fake)      → 출력 mask_s (0 or 1)

    입력: (N, 1536)  N = batch_size × 36 × 36 (patch 수)
    출력: (N, 1)     각 patch의 이상 확률 [0, 1]

    예시:
        입력: (4608, 1536)  → batch=4, 36×36=1296 patches
        출력: (4608, 1)     → 각 patch가 이상일 확률
    """

    def __init__(self, in_planes, n_layers=2, hidden=None):
        """
        Args:
            in_planes: 입력 feature 차원 (= target_embed_dimension = 1536)
            n_layers:  총 레이어 수 (body + tail)
                       n_layers=2 → body 1개 + tail
                       n_layers=3 → body 2개 + tail
            hidden:    hidden layer 차원 (None이면 in_planes에서 점진적 축소)
        """
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()

        # body: (n_layers-1)개의 Linear+BN+LeakyReLU 블록
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module(
                'block%d' % (i + 1),
                torch.nn.Sequential(
                    torch.nn.Linear(_in, _hidden),
                    torch.nn.BatchNorm1d(_hidden),
                    torch.nn.LeakyReLU(0.2),
                )
            )

        # tail: 최종 이진 분류 (Sigmoid로 [0,1] 확률 출력)
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(_hidden, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        self.apply(init_weight)

    def forward(self, x):
        """
        입력: (N, 1536)
        출력: (N, 1) 이상 확률 [0, 1]
        """
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    """
    Feature 공간 변환 레이어 (pre_projection).

    역할:
        backbone이 출력하는 feature를 Discriminator에 적합한 공간으로 변환.
        backbone은 고정(frozen)이므로, 이 레이어를 통해 feature 공간을
        학습 목적에 맞게 조정.

    구조 (pre_proj=1, in_planes=1536):
        Linear(1536 → 1536)  ← 단순 선형 변환 1개

    입력: (N, 1536)
    출력: (N, 1536)

    왜 필요한가?
        backbone feature를 직접 Discriminator에 넣으면 backbone의
        사전학습 feature 공간과 이상 탐지 공간이 달라 학습이 어려움.
        Projection이 이를 중간에서 조율.
    """

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()

        for i in range(n_layers):
            _in  = in_planes if i == 0 else out_planes
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            # 중간 레이어에만 활성화 함수 추가 (마지막 레이어 제외)
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):
        """
        입력: (N, 1536)
        출력: (N, 1536)
        """
        x = self.layers(x)
        return x


class PatchMaker:
    """
    Feature map을 patch 단위로 분해하고 score를 복원하는 유틸리티.

    핵심 역할:
        1. patchify:       feature map (B, C, H, W) → patch features (B×H×W, C, k, k)
                           각 위치의 k×k 이웃을 하나의 patch로 묶음
        2. unpatch_scores: Discriminator 출력 (B×H×W, 1) → (B, H×W, 1)
        3. score:          (B, H×W, 1) → (B,) 이미지 단위 최댓값

    설정값 (MPDD 기준):
        patchsize=3, stride=1
        → padding=1로 spatial 크기 유지 (36×36 → 36×36)
        → 각 patch는 3×3 이웃 포함 → 원본의 24×24 영역 커버
    """

    def __init__(self, patchsize, top_k=0, stride=None):
        """
        Args:
            patchsize: patch 크기 (k), 기본값 3
            top_k:     미사용 (향후 top-k score 평균 구현 예정)
            stride:    sliding stride, 기본값=patchsize (=1로 설정됨)
        """
        self.patchsize = patchsize
        self.stride    = stride
        self.top_k     = top_k

    def patchify(self, features, return_spatial_info=False):
        """
        Feature map을 sliding window patch로 분해.

        torch.nn.Unfold를 사용하여 각 위치의 k×k 이웃을 추출.
        padding=(k-1)/2 설정으로 입력과 출력 spatial 크기 동일.

        Args:
            features: (B, C, H, W) feature map
                      예: (16, 512, 36, 36) layer2 feature
            return_spatial_info: True이면 patch 격자 크기도 반환

        반환:
            unfolded_features: (B, H×W, C, k, k)
                               예: (16, 1296, 512, 3, 3)
                               B=16, 1296=36×36, 512=채널, 3×3=이웃
            number_of_total_patches: [H, W] = [36, 36]
                                     → patch score를 36×36 격자로 복원할 때 사용

        수용 영역(receptive field):
            stride=8 (layer2까지 누적) × patchsize=3 = 24픽셀
            → 각 patch score 1개가 원본의 24×24 픽셀 정보를 담음
        """
        padding = int((self.patchsize - 1) / 2)  # k=3이면 padding=1
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1,
        )
        unfolded_features = unfolder(features)  # (B, C×k×k, H×W)

        # patch 격자 크기 계산: padding으로 인해 입력과 동일한 H, W
        # 공식: n_patches = (s + 2*padding - (k-1) - 1) / stride + 1
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        # 예: s=36, padding=1, k=3, stride=1 → n_patches = 36

        # reshape: (B, C×k×k, H×W) → (B, C, k, k, H×W) → (B, H×W, C, k, k)
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        # 최종: (B, H×W, C, k, k) = (16, 1296, 512, 3, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        """
        Discriminator 출력을 배치 차원으로 복원.

        Args:
            x:         (B×H×W, 1) Discriminator 출력
                       예: (20736, 1) = 16×1296
            batchsize: B = 16

        반환: (B, H×W, 1) = (16, 1296, 1)
              → 이후 reshape으로 (B, 36, 36)으로 변환
        """
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        """
        Patch scores에서 이미지 단위 score 계산 (max pooling).

        현재 방식: 1296개 patch 중 최댓값 1개를 image score로 사용.

        Args:
            x: (B, H×W, 1) = (16, 1296, 1)

        반환: (B,) = (16,) 이미지당 최대 이상 score

        한계:
            국소 고강도 결함에는 유리하지만
            광범위 저강도 결함을 과소평가할 수 있음.
            → top-k 평균, CNN 분류 등으로 개선 가능.
        """
        x = x[:, :, 0]                    # (B, H×W) 마지막 차원 제거
        x = torch.max(x, dim=1).values    # (B,) 1296개 중 최댓값
        return x
