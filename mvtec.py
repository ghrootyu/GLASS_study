# =============================================================================
# MVTecDataset
# 역할: 학습/테스트 이미지를 로드하고, 학습 시 Perlin+DTD 합성 이상(LAS)을 생성
#
# 전체 흐름:
#   __init__ → 경로/변환 설정
#   get_image_data → 이미지·마스크 경로 수집
#   __getitem__ → 이미지 로드 + 학습 시 합성 이상 생성
# =============================================================================

from torchvision import transforms
from perlin import perlin_mask
from enum import Enum

import numpy as np
import pandas as pd

import PIL
import torch
import os
import glob

_CLASSNAMES = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]

# ImageNet 정규화 값 (WideResNet50이 ImageNet으로 사전학습되었으므로 동일 통계값 사용)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST  = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    MVTec / MPDD 등 MVTec 구조를 따르는 데이터셋 로더.

    디렉토리 구조 가정:
        {source}/{classname}/train/good/*.png
        {source}/{classname}/test/good/*.png
        {source}/{classname}/test/{defect}/*.png
        {source}/{classname}/ground_truth/{defect}/*.png  (테스트 GT 마스크)
        {source}/fg_mask/{classname}/*.png                (옵션: foreground 마스크)
    """

    def __init__(
            self,
            source,                              # 데이터셋 루트 경로 (e.g. "/data/MPDD")
            anomaly_source_path='/root/dataset/dtd/images',  # DTD 텍스처 경로 (LAS 합성 재료)
            dataset_name='mvtec',                # 데이터셋 이름 (excel 저장 시 사용)
            classname='leather',                 # 학습할 클래스명 (e.g. "bracket_black")
            resize=288,                          # 이미지 리사이즈 크기
            imagesize=288,                       # 최종 크롭 크기
            split=DatasetSplit.TRAIN,            # TRAIN or TEST
            rotate_degrees=0,                    # 데이터 증강: 회전 범위
            translate=0,                         # 데이터 증강: 이동 비율
            brightness_factor=0,                 # 데이터 증강: 밝기
            contrast_factor=0,                   # 데이터 증강: 대비
            saturation_factor=0,                 # 데이터 증강: 채도
            gray_p=0,                            # 데이터 증강: 그레이스케일 확률
            h_flip_p=0,                          # 데이터 증강: 수평 반전 확률
            v_flip_p=0,                          # 데이터 증강: 수직 반전 확률
            distribution=0,                      # 분포 판별 모드 (0: 파일에서 읽기, 1: FFT 자동 판별)
            mean=0.5,                            # 합성 이상 혼합 비율 평균 (beta 샘플링)
            std=0.1,                             # 합성 이상 혼합 비율 표준편차
            fg=0,                                # foreground 마스크 사용 여부 (0: 미사용, 1: 사용)
            rand_aug=1,                          # DTD 텍스처 랜덤 증강 여부
            downsampling=8,                      # Perlin 마스크 다운샘플 비율 (feature map stride와 일치)
            scale=0,                             # 데이터 증강: 스케일 범위
            batch_size=8,
            **kwargs,
    ):
        super().__init__()
        self.source       = source
        self.split        = split
        self.batch_size   = batch_size
        self.distribution = distribution
        self.mean         = mean
        self.std          = std
        self.fg           = fg
        self.rand_aug     = rand_aug
        self.downsampling = downsampling
        self.classname    = classname
        self.dataset_name = dataset_name

        # distribution==1 이면 FFT 분석용으로 리스트 형태 resize 사용
        self.resize   = resize if self.distribution != 1 else [resize, resize]
        self.imgsize  = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)  # 모델 입력 shape

        # toothbrush, wood는 종횡비가 달라 resize를 조정 (MVTec 특수 처리)
        if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            self.resize = round(self.imgsize * 329 / 288)

        # ── foreground 마스크 사용 여부 결정 ─────────────────────────────────
        # fg=0: 전체 이미지에 Perlin 합성 (배경 포함)
        # fg=1: fg_mask 폴더의 마스크로 물체 영역에만 Perlin 합성
        # fg=2: excel 파일을 읽어 클래스별로 자동 결정
        xlsx_path = './datasets/excel/' + self.dataset_name + '_distribution.xlsx'
        if self.fg == 2:
            try:
                df = pd.read_excel(xlsx_path)
                self.class_fg = df.loc[
                    df['Class'] == self.dataset_name + '_' + classname, 'Foreground'
                ].values[0]
            except:
                self.class_fg = 1
        elif self.fg == 1:
            self.class_fg = 1
        else:
            self.class_fg = 0

        # 이미지/마스크 경로 수집
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # ── DTD 텍스처 경로 수집 ──────────────────────────────────────────────
        # DTD: Describable Textures Dataset (47종 텍스처, 각 120장)
        # 학습 시 정상 이미지와 혼합하여 합성 이상(LAS)을 만드는 재료로 사용
        # 예: dtd/images/banded/banded_0001.jpg, dtd/images/cracked/cracked_0001.jpg, ...
        self.anomaly_source_paths = sorted(
            glob.glob(anomaly_source_path + "/*/*.jpg")
        )

        # ── 이미지 변환 파이프라인 ────────────────────────────────────────────
        # 입력: PIL Image (H, W, 3)
        # 출력: Tensor (3, 288, 288), ImageNet 정규화 적용
        self.transform_img = transforms.Compose([
            transforms.Resize(self.resize),                          # 288×288로 리사이즈
            transforms.ColorJitter(                                  # 색상 증강 (기본값 0 = 미적용)
                brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),              # 수평 반전
            transforms.RandomVerticalFlip(v_flip_p),                # 수직 반전
            transforms.RandomGrayscale(gray_p),                     # 그레이스케일
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(self.imgsize),                    # 중앙 크롭
            transforms.ToTensor(),                                   # [0,255] → [0,1]
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet 정규화
        ])

        # ── 마스크 변환 파이프라인 ────────────────────────────────────────────
        # 마스크는 색상 증강 없이 resize + crop + tensor 변환만 수행
        # 입력: PIL Image (grayscale), 출력: Tensor (1, 288, 288), 값 범위 [0, 1]
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ])

    def rand_augmenter(self):
        """
        DTD 텍스처 이미지에 적용할 랜덤 증강 조합을 생성.

        9가지 증강 중 3개를 무작위로 선택하여 조합.
        이를 통해 합성 이상의 외관 다양성을 높임 → 모델의 일반화 성능 향상.

        반환: transforms.Compose (DTD 이미지에 적용할 변환 파이프라인)
        """
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        # 9개 중 3개 무작위 선택 (중복 없음)
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = transforms.Compose([
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform_aug

    def __getitem__(self, idx):
        """
        데이터셋의 idx번째 샘플을 반환.

        ── TRAIN 모드 반환값 ────────────────────────────────────────────────
        {
            "image"    : (3, 288, 288) 정상 이미지 (정규화됨)
            "aug"      : (3, 288, 288) 합성 이상 이미지 (Perlin+DTD 블렌딩)
            "mask_s"   : (36, 36) Perlin 마스크 (feature map 크기, 1=이상 영역)
            "mask_gt"  : (1, 288, 288) zeros (학습 시 GT 없음)
            "is_anomaly": 0 (학습 이미지는 항상 정상)
            "image_path": 이미지 파일 경로
        }

        ── TEST 모드 반환값 ─────────────────────────────────────────────────
        {
            "image"    : (3, 288, 288) 테스트 이미지
            "aug"      : torch.tensor([1]) (테스트 시 미사용)
            "mask_s"   : torch.tensor([1]) (테스트 시 미사용)
            "mask_gt"  : (1, 288, 288) GT 마스크 (결함 있으면 실제 마스크, 정상이면 zeros)
            "is_anomaly": 0(정상) or 1(이상)
            "image_path": 이미지 파일 경로
        }
        """
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        # ── 1. 정상 이미지 로드 및 변환 ──────────────────────────────────────
        # 학습 시: train/good/*.png 이미지
        # 결과: (3, 288, 288) ImageNet 정규화된 텐서
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # 기본값 초기화 (테스트 시 aug, mask_s는 사용 안 함)
        mask_fg = mask_s = aug_image = torch.tensor([1])

        if self.split == DatasetSplit.TRAIN:
            # ── 2. DTD 텍스처 이미지 로드 (LAS 합성 재료) ───────────────────
            # anomaly_source_paths: DTD 이미지 경로 목록 (~5640장)
            # 매 샘플마다 무작위로 1장 선택
            # 예: dtd/images/cracked/cracked_0042.jpg
            aug = PIL.Image.open(
                np.random.choice(self.anomaly_source_paths)
            ).convert("RGB")

            # DTD 이미지에 랜덤 증강 적용 (다양한 텍스처 패턴 생성)
            # rand_aug=1 이면 9가지 증강 중 3개 랜덤 조합
            # rand_aug=0 이면 정상 이미지와 동일한 변환만 적용
            if self.rand_aug:
                transform_aug = self.rand_augmenter()
                aug = transform_aug(aug)          # (3, 288, 288)
            else:
                aug = self.transform_img(aug)     # (3, 288, 288)

            # ── 3. Foreground 마스크 로드 (옵션) ────────────────────────────
            # fg=1이면 물체 영역(foreground)에만 Perlin 합성을 적용
            # fg=0이면 mask_fg = 1 (스칼라) → 전체 이미지에 합성
            # 예: fg_mask/bracket_black/000.png → 브라켓 부분만 255, 배경은 0
            if self.class_fg:
                fgmask_path = (
                    image_path.split(classname)[0]
                    + 'fg_mask/' + classname + '/'
                    + os.path.split(image_path)[-1]
                )
                mask_fg = PIL.Image.open(fgmask_path)
                # transform_mask 후 첫 번째 채널만 사용, ceil로 0/1 이진화
                # 결과: (288, 288) 텐서, 0=배경 / 1=foreground
                mask_fg = torch.ceil(self.transform_mask(mask_fg)[0])

            # ── 4. Perlin 노이즈 마스크 생성 ────────────────────────────────
            # Perlin 노이즈로 자연스러운 형태의 이상 영역 마스크를 생성
            #
            # 입력:
            #   image.shape = (3, 288, 288)
            #   feat_size   = 288 // 8 = 36  (feature map 크기와 일치)
            #   min=0, max=6                  (Perlin 스케일 범위)
            #   mask_fg                       (foreground 제한 마스크)
            #
            # 출력:
            #   mask_all[0] = mask_s: (36, 36)   feature map 크기의 이상 마스크
            #                         1=이상 영역, 0=정상 영역
            #                         → Discriminator 학습의 pixel-level label로 사용
            #   mask_all[1] = mask_l: (288, 288)  원본 크기의 이상 마스크
            #                         → 이미지 블렌딩에 사용
            mask_all = perlin_mask(
                image.shape,
                self.imgsize // self.downsampling,  # 36
                0, 6,
                mask_fg,
                1,
            )
            mask_s = torch.from_numpy(mask_all[0])  # (36, 36)
            mask_l = torch.from_numpy(mask_all[1])  # (288, 288)

            # ── 5. 합성 이상 이미지 생성 (LAS: Local Anomaly Synthesis) ─────
            # beta: 원본 이미지의 잔류 비율 (정규분포 샘플링, 0.2~0.8로 클리핑)
            # 예: beta=0.3이면 이상 영역에서 70% DTD + 30% 원본 혼합
            beta = np.random.normal(loc=self.mean, scale=self.std)
            beta = np.clip(beta, .2, .8)

            # 블렌딩 공식:
            #   정상 영역 (mask_l==0): 원본 이미지 그대로
            #   이상 영역 (mask_l==1): (1-beta)*DTD + beta*원본
            #
            # 예시 (beta=0.3):
            #   이상 영역 = 0.7 * dtd_texture + 0.3 * original
            #   → 원본 구조를 30% 유지하면서 텍스처를 교체 → 자연스러운 결함 시뮬레이션
            aug_image = (
                image * (1 - mask_l)                    # 정상 영역: 원본
                + (1 - beta) * aug * mask_l             # 이상 영역: DTD 성분
                + beta * image * mask_l                  # 이상 영역: 원본 잔류
            )
            # aug_image shape: (3, 288, 288)

        # ── 6. 테스트 GT 마스크 로드 ─────────────────────────────────────────
        if self.split == DatasetSplit.TEST and mask_path is not None:
            # 결함 이미지: 실제 GT 마스크 로드
            # 예: ground_truth/hole/000_mask.png → 결함 위치 255, 나머지 0
            # 변환 후: (1, 288, 288), 값 범위 [0, 1]
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            # 정상 이미지 또는 학습 모드: GT 마스크 없음 → zeros
            mask_gt = torch.zeros([1, *image.size()[1:]])  # (1, 288, 288)

        return {
            "image"     : image,       # (3, 288, 288) 원본(정상) 이미지
            "aug"       : aug_image,   # (3, 288, 288) 합성 이상 이미지 (학습 시만 유효)
            "mask_s"    : mask_s,      # (36, 36) Perlin 마스크 (학습 시만 유효)
            "mask_gt"   : mask_gt,     # (1, 288, 288) GT 마스크 (테스트 시만 유효)
            "is_anomaly": int(anomaly != "good"),  # 0=정상, 1=이상
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        """
        데이터셋의 이미지·마스크 경로를 수집하여 반환.

        ── TRAIN 모드 ────────────────────────────────────────────────────────
        train/good/ 폴더의 이미지만 수집 (정상 이미지만 학습에 사용)

        예시 (bracket_black):
            imgpaths_per_class = {
                "bracket_black": {
                    "good": ["MPDD/bracket_black/train/good/000.png", ...]
                }
            }
            data_to_iterate = [
                ["bracket_black", "good", "MPDD/.../000.png", None],
                ["bracket_black", "good", "MPDD/.../001.png", None],
                ...
            ]

        ── TEST 모드 ─────────────────────────────────────────────────────────
        test/good/ + test/{defect}/ 폴더의 이미지 모두 수집
        결함 이미지는 ground_truth/{defect}/ 마스크와 1:1 매칭

        예시 (bracket_black):
            data_to_iterate = [
                ["bracket_black", "good",     ".../test/good/000.png",     None],
                ["bracket_black", "hole",     ".../test/hole/000.png",     ".../ground_truth/hole/000_mask.png"],
                ["bracket_black", "scratches",".../test/scratches/000.png",".../ground_truth/scratches/000_mask.png"],
                ...
            ]

        반환:
            imgpaths_per_class: dict {classname: {anomaly_type: [image_paths]}}
            data_to_iterate: list of [classname, anomaly, image_path, mask_path]
        """
        imgpaths_per_class  = {}
        maskpaths_per_class = {}

        # 예: MPDD/bracket_black/train 또는 MPDD/bracket_black/test
        classpath = os.path.join(self.source, self.classname, self.split.value)
        # 예: MPDD/bracket_black/ground_truth
        maskpath  = os.path.join(self.source, self.classname, "ground_truth")

        # 해당 split 폴더의 서브디렉토리 목록 = anomaly 종류
        # TRAIN: ["good"]
        # TEST:  ["good", "hole", "scratches"] (bracket_black 예시)
        anomaly_types = os.listdir(classpath)

        imgpaths_per_class[self.classname]  = {}
        maskpaths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            anomaly_path  = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))

            # 이미지 경로 저장
            imgpaths_per_class[self.classname][anomaly] = [
                os.path.join(anomaly_path, x) for x in anomaly_files
            ]

            # 마스크 경로: 테스트 + 결함 종류일 때만 수집
            # good 이미지는 GT 마스크 없음 → None
            if self.split == DatasetSplit.TEST and anomaly != "good":
                anomaly_mask_path  = os.path.join(maskpath, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                maskpaths_per_class[self.classname][anomaly] = [
                    os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                ]
            else:
                maskpaths_per_class[self.classname]["good"] = None

        # data_to_iterate 구성: [classname, anomaly, image_path, mask_path]
        # 이미지와 마스크는 정렬 후 인덱스로 1:1 매칭
        # 예: 000.png ↔ 000_mask.png (sorted 순서 기준)
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)  # 정상 or 학습: 마스크 없음
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
