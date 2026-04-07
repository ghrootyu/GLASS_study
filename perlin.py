# =============================================================================
# Perlin 노이즈 기반 이상 마스크 생성 (LAS: Local Anomaly Synthesis)
#
# 역할:
#   자연스러운 형태의 이상 영역 마스크를 생성하여
#   DTD 텍스처를 정상 이미지의 특정 위치에 합성할 때 사용.
#
# 핵심 아이디어:
#   랜덤 사각형이나 원이 아닌 Perlin 노이즈를 사용함으로써
#   실제 결함처럼 불규칙하고 자연스러운 형태의 마스크를 생성.
#
# 사용 위치: mvtec.py __getitem__ → perlin_mask() 호출
# =============================================================================

import imgaug.augmenters as iaa
import numpy as np
import torch
import math


def generate_thr(img_shape, min=0, max=4):
    """
    단일 Perlin 노이즈 이진 마스크를 생성.

    Perlin 노이즈는 연속적인 부드러운 패턴을 만들어내는 절차적 노이즈.
    임계값 0.5를 기준으로 이진화하여 0/1 마스크 생성.

    Args:
        img_shape: 이미지 shape (C, H, W) - H, W만 사용
        min, max:  Perlin 스케일 범위. 클수록 거친 패턴, 작을수록 세밀한 패턴
                   예: scale=2^1=2이면 작은 블롭, 2^4=16이면 큰 덩어리

    반환: (H, W) numpy array, 값 0 or 1
    예시 (H=288, W=288):
        [[0, 0, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0], ...]  ← 불규칙한 블롭 형태
    """
    min_perlin_scale = min
    max_perlin_scale = max

    # 2^min ~ 2^max 범위에서 x, y 방향 스케일 독립적으로 랜덤 선택
    # 예: perlin_scalex=4 (2^2), perlin_scaley=8 (2^3) → 타원형 패턴
    perlin_scalex = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scaley = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)

    # (H, W) 크기의 Perlin 노이즈 생성, 값 범위 [-√2, √2]
    perlin_noise_np = rand_perlin_2d_np(
        (img_shape[1], img_shape[2]),
        (perlin_scalex, perlin_scaley)
    )

    # 랜덤 회전으로 패턴 다양성 증가 (-90° ~ 90°)
    perlin_noise_np = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(
        image=perlin_noise_np
    )

    # 임계값 0.5로 이진화: 0.5 초과 → 1 (이상 영역), 이하 → 0 (정상 영역)
    perlin_thr = np.where(
        perlin_noise_np > 0.5,
        np.ones_like(perlin_noise_np),
        np.zeros_like(perlin_noise_np)
    )
    return perlin_thr  # (H, W), dtype float64


def perlin_mask(img_shape, feat_size, min, max, mask_fg, flag=0):
    """
    최종 Perlin 마스크 생성 (두 크기: feature map용 + 원본용).

    두 개의 Perlin 마스크를 조합하여 더 다양한 형태의 이상 영역 생성:
      - AND(교집합): 두 마스크 모두 1인 영역 → 작고 국소적인 결함
      - OR(합집합):  둘 중 하나라도 1인 영역 → 크고 넓은 결함
      - 단독 사용:   하나만 사용 → 중간 크기

    Foreground 마스크(mask_fg)를 곱하여 물체 영역에만 합성.

    Args:
        img_shape: 이미지 shape (3, 288, 288)
        feat_size: feature map 크기 = 288 // 8 = 36
                   → mask_s는 (36, 36) Discriminator의 pixel-level label로 사용
        min, max:  Perlin 스케일 범위 (0~6)
        mask_fg:   foreground 마스크 (288, 288) 또는 스칼라 1 (fg=0일 때)
        flag:      0이면 mask_s만, 1이면 (mask_s, mask_l) 반환

    반환 (flag=1):
        mask_s: (36, 36) feature map 크기 마스크 → Discriminator label
        mask_l: (288, 288) 원본 크기 마스크 → 이미지 블렌딩용

    예시:
        mask_s = [[0,0,1,1,...],   ← 36×36, Discriminator가 이 위치를 이상으로 학습
                  [0,1,1,0,...], ...]
        mask_l = [[0,0,...,1,1,...,0],  ← 288×288, aug_image 생성 시 이 영역에 DTD 적용
                  ...]
    """
    mask = np.zeros((feat_size, feat_size))  # 유효한 마스크가 생길 때까지 반복

    # mask가 모두 0이면 (합성 영역이 없으면) 재시도
    while np.max(mask) == 0:
        # 두 개의 독립적인 Perlin 마스크 생성
        perlin_thr_1 = generate_thr(img_shape, min, max)  # (288, 288)
        perlin_thr_2 = generate_thr(img_shape, min, max)  # (288, 288)

        temp = torch.rand(1).numpy()[0]  # 0~1 균등 난수

        if temp > 2 / 3:
            # 33% 확률: OR 연산 → 넓은 이상 영역
            # 두 마스크 중 하나라도 1이면 1
            perlin_thr = perlin_thr_1 + perlin_thr_2
            perlin_thr = np.where(
                perlin_thr > 0,
                np.ones_like(perlin_thr),
                np.zeros_like(perlin_thr)
            )
        elif temp > 1 / 3:
            # 33% 확률: AND 연산 → 작은 이상 영역
            # 두 마스크 모두 1인 영역만 1
            perlin_thr = perlin_thr_1 * perlin_thr_2
        else:
            # 33% 확률: 단독 사용 → 중간 크기
            perlin_thr = perlin_thr_1

        perlin_thr = torch.from_numpy(perlin_thr)  # (288, 288) tensor

        # foreground 마스크 적용: 물체 영역(=1)에만 이상 합성
        # mask_fg=1 (스칼라)이면 전체 영역에 합성
        # mask_fg=(288,288)이면 물체 위치에만 합성
        perlin_thr_fg = perlin_thr * mask_fg  # (288, 288)

        # ── feature map 크기(36×36)로 다운샘플 ──────────────────────────────
        # Discriminator가 보는 단위가 feature map 픽셀이므로
        # mask_s도 feature map 크기(36×36)로 맞춰야 함
        # max_pool: 8×8 구간 중 하나라도 이상(=1)이면 해당 feature 위치를 이상으로 표시
        down_ratio_y = int(img_shape[1] / feat_size)  # 288 // 36 = 8
        down_ratio_x = int(img_shape[2] / feat_size)  # 288 // 36 = 8

        mask_ = perlin_thr_fg  # (288, 288) 원본 크기 마스크 (mask_l용)
        # max_pool2d로 8×8 → 1로 축소: 이상 영역이 하나라도 있으면 1
        mask = torch.nn.functional.max_pool2d(
            perlin_thr_fg.unsqueeze(0).unsqueeze(0),  # (1, 1, 288, 288)
            (down_ratio_y, down_ratio_x)               # kernel=(8,8)
        ).float()
        mask = mask.numpy()[0, 0]  # (36, 36)

    # mask_s: (36, 36) feature map 크기 → Discriminator pixel-level label
    mask_s = mask

    if flag != 0:
        # mask_l: (288, 288) 원본 크기 → 이미지 블렌딩용
        mask_l = mask_.numpy()

    if flag == 0:
        return mask_s
    else:
        return mask_s, mask_l


def lerp_np(x, y, w):
    """선형 보간: x + (y-x)*w"""
    return (y - x) * w + x


def rand_perlin_2d_np(shape, res, fade=lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    """
    2D Perlin 노이즈 생성 (numpy 구현).

    Perlin 노이즈 원리:
      1. 그리드 각 꼭짓점에 랜덤 기울기 벡터(gradient) 배치
      2. 각 픽셀 위치에서 4개의 꼭짓점 기울기와의 내적 계산
      3. fade 함수(6t^5-15t^4+10t^3)로 부드럽게 보간
      → 연속적이고 부드러운 노이즈 패턴 생성

    Args:
        shape: 출력 크기 (H, W) e.g. (288, 288)
        res:   Perlin 그리드 해상도 (res_x, res_y) e.g. (4, 8)
               값이 클수록 거친 패턴 (큰 덩어리)

    반환: (H, W) numpy array, 값 범위 [-√2, √2] ≈ [-1.41, 1.41]
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    # 그리드 좌표 생성
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # 각 그리드 꼭짓점에 랜덤 각도 → 기울기 벡터 생성
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0),
        d[1], axis=1
    )
    dot = lambda grad, shift: (
        np.stack(
            (grid[:shape[0], :shape[1], 0] + shift[0],
             grid[:shape[0], :shape[1], 1] + shift[1]),
            axis=-1
        ) * grad[:shape[0], :shape[1]]
    ).sum(axis=-1)

    # 4개 꼭짓점에서의 기울기-거리 내적
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    # fade 함수로 부드러운 보간 (Hermite 보간)
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]),
        lerp_np(n01, n11, t[..., 0]),
        t[..., 1]
    )
