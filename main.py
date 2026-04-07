# =============================================================================
# GLASS 진입점 (Entry Point)
#
# 전체 실행 흐름:
#
#   main() [click group]
#     ├─ net()     → get_glass 팩토리 함수 반환
#     │              (backbone + GLASS 인스턴스 생성 담당)
#     └─ dataset() → get_dataloaders 팩토리 함수 반환
#                    (train/test DataLoader 생성 담당)
#
#   run() [result_callback]
#     ← net()과 dataset()의 반환값을 모아 실제 학습/추론 수행
#     1. get_dataloaders(seed, test) → 클래스별 DataLoader 쌍 생성
#     2. get_glass(imagesize, device) → GLASS 모델 인스턴스 생성
#     3. 클래스별 루프:
#        ├─ test='ckpt' → GLASS.trainer() 학습 실행
#        └─ 학습 완료 → GLASS.tester() 최종 평가 + 결과 CSV 저장
#
# Click chain 구조:
#   click.group(chain=True)로 서브커맨드를 순서대로 실행.
#   "main net ... dataset ..."처럼 한 줄에 여러 서브커맨드 나열 가능.
#   각 서브커맨드는 (key, value) 튜플을 반환 → run()에서 dict로 합산.
#
# 실행 예시 (shell):
#   python main.py --gpu 0 --seed 0 --test ckpt \
#     net -b wideresnet50 -le layer2 -le layer3 \
#         --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
#         --meta_epochs 640 --dsc_layers 2 \
#     dataset --fg 0 --batch_size 16 -d bracket_black \
#         mpdd /path/to/mpdd /path/to/dtd
# =============================================================================

from datetime import datetime

import pandas as pd
import os
import logging
import sys
import click
import torch
import warnings
import backbones
import glass
import utils


# =============================================================================
# main: click group (최상위 옵션 정의)
#
# chain=True: 여러 서브커맨드(net, dataset)를 순서대로 실행하도록 허용.
# 각 서브커맨드의 반환값이 result_callback(run)으로 전달됨.
#
# 옵션:
#   --results_path : 결과 저장 루트 디렉터리 (기본값: "results")
#                    구조: results/{log_project}/{log_group}/{run_name}/
#   --gpu          : 사용할 GPU 번호 (multiple=True → 여러 개 지정 가능)
#                    예: --gpu 0 --gpu 1
#   --seed         : 전역 랜덤 시드 (재현성 보장)
#   --log_group    : 결과 폴더 하위 그룹명 (기본값: "group")
#   --log_project  : 결과 폴더 프로젝트명 (기본값: "project")
#   --run_name     : 실행 이름 (기본값: "test")
#   --test         : 실행 모드
#                    'ckpt' → 학습 후 최적 체크포인트 평가 (기본)
#                    'test' → 학습 없이 기존 체크포인트로 평가만 수행
# =============================================================================
@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="ckpt")
def main(**kwargs):
    pass


# =============================================================================
# net: 모델 구성 서브커맨드
#
# 역할:
#   GLASS 모델을 초기화하는 팩토리 함수(get_glass)를 생성하여 반환.
#   실제 모델 생성은 run()에서 imagesize와 device가 확정된 후 호출됨.
#
# 주요 하이퍼파라미터:
#   -b / --backbone_names          : backbone 이름 (복수 지정 가능)
#                                    예: -b wideresnet50
#                                    ".seed-N" 접미사로 backbone별 시드 지정 가능
#                                    예: -b wideresnet50.seed-42
#   -le / --layers_to_extract_from : feature 추출 레이어 (복수 지정)
#                                    예: -le layer2 -le layer3
#                                    layer2: (B, 512, 36, 36) — 저수준(텍스처)
#                                    layer3: (B, 1024, 18, 18) — 고수준(의미)
#   --pretrain_embed_dimension     : Preprocessing 압축 목표 차원 (기본: 1024)
#                                    MeanMapper가 layer2/layer3 feature를
#                                    이 차원으로 통일하여 두 레이어 합산 가능하게 함
#                                    MPDD 설정에서는 1536으로 높여 정보 손실 최소화
#   --target_embed_dimension       : Aggregator 최종 출력 차원 (기본: 1024)
#                                    Discriminator 입력 차원과 동일해야 함
#   --patchsize                    : PatchMaker 패치 크기 k (기본: 3)
#                                    k=3이면 각 패치가 3×3 이웃 포함
#                                    수용 영역: stride(8) × k(3) = 24 픽셀
#   --meta_epochs                  : 전체 학습 에폭 수 (기본: 640)
#                                    MPDD 설정: 640 (긴 학습으로 수렴 보장)
#   --eval_epochs                  : 몇 에폭마다 평가할지 (기본: 1)
#                                    1이면 매 에폭 평가 (최적 체크포인트 탐색)
#   --dsc_layers                   : Discriminator 레이어 수 (기본: 2)
#                                    2 → body 1개(1536→1024) + tail(1024→1)
#   --dsc_hidden                   : Discriminator 은닉층 차원 (기본: 1024)
#   --dsc_margin                   : GAS 손실의 마진 임계값 (기본: 0.5)
#                                    center에서 radius 이내면 BCE=0 (안전 영역)
#   --pre_proj                     : Projection 레이어 수 (기본: 1)
#                                    1 → Linear(1536→1536) 1개만 사용
#   --mining                       : Hard negative mining 사용 여부 (기본: 1)
#                                    1 → 상위 p% 어려운 샘플만 Focal Loss 적용
#   --noise                        : GAS에서 Gaussian noise 표준편차 (기본: 0.015)
#                                    true_feats에 더하는 노이즈 강도
#                                    너무 크면 너무 쉬운 fake 생성 → 학습 효과 감소
#   --radius                       : GAS hypersphere 반지름 (기본: 0.75)
#                                    gaus_feats를 [r, 2r] shell에 투영
#                                    → 정상 center에서 적당히 떨어진 위치에 배치
#   --p                            : Hard mining 비율 (기본: 0.5)
#                                    전체 fake patch 중 상위 50%만 FocalLoss 적용
#   --lr                           : 학습률 (기본: 0.0001)
#                                    Projection: lr × 1 = 1e-4
#                                    Discriminator: lr × 2 = 2e-4 (더 빠른 학습)
#   --svd                          : 분포 판별 모드 (기본: 0)
#                                    0 = Manifold (복잡한 구조적 결함 — MPDD 기본)
#                                    1 = Hypersphere (균일한 텍스처 결함)
#                                    datasets/excel/*.xlsx에 자동 저장됨
#   --step                         : Gradient Ascent 반복 횟수 (기본: 20)
#                                    20회 gradient ascent로 gaus_feats를 결정경계 방향으로 이동
#   --limit                        : 에폭당 최대 처리 샘플 수 (기본: 392)
#                                    392 = 49배치 × 8샘플 (batch_size=8 기준)
#                                    배치 수로 에폭 길이를 고정하여 학습 속도 균등화
# =============================================================================
@main.command("net")
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=640)
@click.option("--eval_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--pre_proj", type=int, default=1)
@click.option("--mining", type=int, default=1)
@click.option("--noise", type=float, default=0.015)
@click.option("--radius", type=float, default=0.75)
@click.option("--p", type=float, default=0.5)
@click.option("--lr", type=float, default=0.0001)
@click.option("--svd", type=int, default=0)
@click.option("--step", type=int, default=20)
@click.option("--limit", type=int, default=392)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        meta_epochs,
        eval_epochs,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        train_backbone,
        pre_proj,
        mining,
        noise,
        radius,
        p,
        lr,
        svd,
        step,
        limit,
):
    """
    GLASS 모델 팩토리 함수를 구성하고 반환.

    반환값: ("get_glass", get_glass)
        → run()에서 methods["get_glass"](imagesize, device)로 호출
        → GLASS 인스턴스 리스트 반환

    backbone이 여러 개인 경우:
        backbone마다 독립적인 GLASS 인스턴스를 생성하고 앙상블로 사용 가능.
        MPDD 설정에서는 backbone 1개(-b wideresnet50)만 사용.

    예시:
        get_glass((288, 288), device) 호출 시
        → WideResNet50 backbone 로드
        → GLASS 인스턴스 생성 및 load() 호출 (Discriminator, Projection 초기화)
        → [glass_inst] 반환 (리스트)
    """
    backbone_names = list(backbone_names)

    # ── backbone이 여러 개인 경우 레이어 설정 복제 ────────────────────────────
    # 예: backbone_names=["resnet50", "wideresnet50"] → 각각 ["layer2", "layer3"] 적용
    # backbone이 1개이면 [layers_to_extract_from] 그대로 사용
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_glass(input_shape, device):
        """
        GLASS 인스턴스를 생성하고 초기화하는 팩토리 함수.

        Args:
            input_shape: 입력 이미지 크기 (imagesize, imagesize) = (288, 288)
                         DataLoader의 dataset.imagesize에서 가져옴
            device:      연산 디바이스 (GPU/CPU)

        반환: [GLASS 인스턴스, ...] — backbone 수만큼의 GLASS 인스턴스 리스트

        처리 흐름:
            1. backbone 이름에서 seed 파싱 (예: "wideresnet50.seed-42" → name="wideresnet50", seed=42)
            2. backbones.load(name) → WideResNet50 사전학습 모델 로드 (ImageNet 가중치)
            3. glass.GLASS(device) 인스턴스 생성
            4. glass_inst.load(...) → Discriminator, Projection, PatchMaker, etc. 초기화
            5. glass_inst.to(device) → GPU로 이동
        """
        glasses = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            # ── backbone seed 파싱 ──────────────────────────────────────────
            # "wideresnet50.seed-42" → backbone_name="wideresnet50", backbone_seed=42
            # seed가 없으면 backbone_seed=None → 전역 seed 사용
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])

            # ── backbone 로드 ────────────────────────────────────────────────
            # backbones.load("wideresnet50") → torchvision WideResNet50_V1 (ImageNet 사전학습)
            # 가중치 고정(frozen): NetworkFeatureAggregator에서 no_grad로 순전파
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            # ── GLASS 인스턴스 생성 및 초기화 ───────────────────────────────
            # glass.GLASS(device): 모델 컨테이너 생성
            # load(): 아래 모듈들을 초기화
            #   - NetworkFeatureAggregator: layer2, layer3 hook 등록
            #   - Preprocessing: MeanMapper × 2 (각 레이어별 차원 통일)
            #   - Aggregator: 2 레이어 평균 → 최종 feature
            #   - Discriminator: 이진 분류기 (1536 → 1)
            #   - Projection: feature 공간 변환 (1536 → 1536)
            #   - PatchMaker: patch 분해/복원 유틸리티
            #   - Optimizer: Discriminator(lr×2) + Projection(lr×1) AdamW
            glass_inst = glass.GLASS(device)
            glass_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                mining=mining,
                noise=noise,
                radius=radius,
                p=p,
                lr=lr,
                svd=svd,
                step=step,
                limit=limit,
            )
            glasses.append(glass_inst.to(device))
        return glasses

    # click chain 구조: (key, factory_fn) 튜플 반환
    # run()에서 methods = dict(net결과, dataset결과)로 합산
    # methods["get_glass"](imagesize, device) 로 호출
    return "get_glass", get_glass


# =============================================================================
# dataset: 데이터셋 구성 서브커맨드
#
# 역할:
#   train/test DataLoader를 생성하는 팩토리 함수(get_dataloaders)를 반환.
#   실제 DataLoader 생성은 run()에서 seed와 test 모드가 확정된 후 호출됨.
#
# 인수 (Arguments) — 위치 기반, 필수:
#   name       : 데이터셋 종류 ("mvtec", "visa", "mpdd", "wfdd")
#                내부적으로 _DATASETS 딕셔너리에서 데이터셋 클래스 매핑
#                mpdd와 wfdd는 mvtec 클래스를 재사용 (폴더 구조 동일)
#   data_path  : 데이터셋 루트 경로 (존재해야 함)
#                예: /home/ghyu/GLASS_WORK/dataset/MPDD
#   aug_path   : DTD 텍스처 이미지 경로 (LAS 합성에 사용)
#                예: /home/ghyu/GLASS_WORK/dataset/dtd/images
#
# 옵션:
#   -d / --subdatasets    : 학습/평가할 클래스 이름 (복수 지정 가능)
#                           예: -d bracket_black -d bracket_brown
#   --batch_size          : 배치 크기 (기본: 8, MPDD: 16)
#   --num_workers         : DataLoader 워커 수 (기본: 16)
#                           WSL2 환경에서는 4로 낮추는 것 권장
#   --resize              : 로드 시 리사이즈 크기 (기본: 288)
#                           WideResNet50의 stride 합산: 288/8=36 패치격자
#   --imagesize           : 최종 크롭/패딩 크기 (기본: 288, resize와 동일하게 사용)
#   --rotate_degrees      : 랜덤 회전 각도 (기본: 0, 비활성)
#   --translate           : 랜덤 평행이동 비율 (기본: 0.0, 비활성)
#   --scale               : 랜덤 스케일 변동 (기본: 0.0, 비활성)
#   --brightness/contrast/saturation/gray : 색상 증강 (기본: 0.0, 비활성)
#   --hflip / --vflip     : 수평/수직 뒤집기 확률 (기본: 0.0, 비활성)
#   --distribution        : 분포 판별 모드 강제 설정
#                           0=자동판별, 1=Manifold, 2=Hypersphere
#   --mean / --std        : GAS Gaussian 노이즈 분포 파라미터
#                           mean=0.5, std=0.1 → N(0.5, 0.1²)
#   --fg                  : foreground 마스크 사용 여부
#                           1 → 사전 생성된 fg_mask PNG 파일 사용
#                               물체 영역에만 Perlin 합성 적용
#                           0 → fg_mask 미사용 (전체 영역에 합성)
#                               배경이 복잡한 경우 0 권장 (MPDD: --fg 0)
#   --rand_aug            : 랜덤 증강 적용 여부 (기본: 1)
#   --downsampling        : feature map 다운샘플 비율 (기본: 8)
#                           layer2 stride와 동일하게 설정: 288/8=36 패치격자
#   --augment             : is_flag, 추가 증강 활성화
# =============================================================================
@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("aug_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=16, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--distribution", default=0, type=int)
@click.option("--mean", default=0.5, type=float)
@click.option("--std", default=0.1, type=float)
@click.option("--fg", default=1, type=int)
@click.option("--rand_aug", default=1, type=int)
@click.option("--downsampling", default=8, type=int)
@click.option("--augment", is_flag=True)
def dataset(
        name,
        data_path,
        aug_path,
        subdatasets,
        batch_size,
        resize,
        imagesize,
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        distribution,
        mean,
        std,
        fg,
        rand_aug,
        downsampling,
        augment,
):
    """
    DataLoader 팩토리 함수를 구성하고 반환.

    반환값: ("get_dataloaders", get_dataloaders)
        → run()에서 methods["get_dataloaders"](seed, test)로 호출
        → 클래스별 {"training": train_dl, "testing": test_dl} 딕셔너리 리스트 반환

    데이터셋 매핑:
        _DATASETS = {
            "mvtec": ["datasets.mvtec", "MVTecDataset"],
            "visa":  ["datasets.visa",  "VisADataset"],
            "mpdd":  ["datasets.mvtec", "MVTecDataset"],  ← MVTecDataset 재사용
            "wfdd":  ["datasets.mvtec", "MVTecDataset"],  ← MVTecDataset 재사용
        }
        MPDD는 MVTec과 동일한 폴더 구조를 가지므로 MVTecDataset 그대로 사용.

    예시 (subdatasets=["bracket_black", "bracket_brown"]):
        get_dataloaders(seed=0, test='ckpt') 반환:
        [
            {"training": DataLoader(bracket_black 학습), "testing": DataLoader(bracket_black 테스트)},
            {"training": DataLoader(bracket_brown 학습), "testing": DataLoader(bracket_brown 테스트)},
        ]
    """
    # ── 데이터셋 클래스 동적 임포트 ─────────────────────────────────────────
    # name="mpdd" → dataset_info = ["datasets.mvtec", "MVTecDataset"]
    # __import__("datasets.mvtec", fromlist=["MVTecDataset"]) → datasets.mvtec 모듈
    # dataset_library.MVTecDataset → 클래스 참조
    _DATASETS = {
        "mvtec": ["datasets.mvtec", "MVTecDataset"],
        "visa":  ["datasets.visa",  "VisADataset"],
        "mpdd":  ["datasets.mvtec", "MVTecDataset"],
        "wfdd":  ["datasets.mvtec", "MVTecDataset"],
    }
    dataset_info    = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, test, get_name=name):
        """
        클래스별 train/test DataLoader 생성.

        Args:
            seed:     랜덤 시드 (재현성)
            test:     'ckpt' → 학습 모드 (train+test DataLoader 생성)
                      기타   → 추론 모드 (test DataLoader만 생성)
            get_name: 데이터셋 이름 (DataLoader.name 설정용)

        반환: list of dict
              각 원소: {"training": DataLoader, "testing": DataLoader}
              원소 수 = len(subdatasets)

        DataLoader 설정:
            test_dataloader:
                shuffle=False (순서 보장 → 결과 재현성)
                pin_memory=True (GPU 전송 가속)
                prefetch_factor=2 (다음 배치 미리 로드)

            train_dataloader:
                shuffle=True (에폭마다 순서 섞음)
                → 동일 배치가 반복되는 과적합 방지

        test='ckpt'일 때 train_dataset 생성:
            split=TRAIN → 정상 이미지만 로드 (이상 이미지 제외)
            LAS 합성 파라미터 전달:
                fg=0         → 전체 영역에 Perlin 합성 (MPDD)
                downsampling=8 → 288/8=36 패치격자 크기 결정
                distribution=0 → Manifold/Hypersphere 자동 판별
                mean=0.5, std=0.1 → GAS 노이즈 분포 파라미터
        """
        dataloaders = []
        for subdataset in subdatasets:

            # ── 테스트 데이터셋 (항상 생성) ──────────────────────────────────
            # split=TEST → test/good + test/{defect_type} 이미지 + 정답 마스크 로드
            # 예: MPDD/bracket_black/test/good/000.png (정상)
            #     MPDD/bracket_black/test/bent/000.png (이상)
            #     MPDD/bracket_black/ground_truth/bent/000_mask.png (정답 마스크)
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,         # 순서 유지: 결과 파일명과 이미지 순서 일치
                num_workers=num_workers,
                prefetch_factor=2,     # 다음 배치 미리 로드 → GPU 유휴시간 감소
                pin_memory=True,       # CPU→GPU 전송 가속 (page-locked memory 사용)
            )

            # DataLoader에 이름 부여 → 결과 저장 경로에 사용
            # 예: "mpdd_bracket_black" → results/.../mpdd_bracket_black/
            test_dataloader.name = get_name + "_" + subdataset

            if test == 'ckpt':
                # ── 학습 데이터셋 (test='ckpt'일 때만 생성) ─────────────────
                # split=TRAIN → train/good 이미지만 로드 (정상 이미지만)
                # GLASS는 one-class 학습: 정상 데이터만으로 Discriminator 훈련
                # 예: MPDD/bracket_black/train/good/*.png
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.TRAIN,
                    seed=seed,
                    rotate_degrees=rotate_degrees,
                    translate=translate,
                    brightness_factor=brightness,
                    contrast_factor=contrast,
                    saturation_factor=saturation,
                    gray_p=gray,
                    h_flip_p=hflip,
                    v_flip_p=vflip,
                    scale=scale,
                    distribution=distribution,
                    mean=mean,
                    std=std,
                    fg=fg,                  # 0=전체합성, 1=foreground만 합성
                    rand_aug=rand_aug,
                    downsampling=downsampling,  # 288/8=36 패치격자
                    augment=augment,
                    batch_size=batch_size,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,          # 에폭마다 순서 섞음 → 과적합 방지
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=True,
                )

                train_dataloader.name = test_dataloader.name
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={len(train_dataset)} test={len(test_dataset)}")
            else:
                # 추론 모드: 학습 없이 기존 체크포인트로 평가만 수행
                # train_dataloader를 test_dataloader로 대체 (trainer()가 호출되지 않으므로 미사용)
                train_dataloader = test_dataloader
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={0} test={len(test_dataset)}")

            dataloader_dict = {
                "training": train_dataloader,
                "testing":  test_dataloader,
            }
            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return "get_dataloaders", get_dataloaders


# =============================================================================
# run: 실제 학습/추론 수행 (result_callback)
#
# 역할:
#   net()과 dataset()이 반환한 팩토리 함수들을 받아 실제 학습/평가 실행.
#   click의 result_callback으로 등록되어 모든 서브커맨드 완료 후 자동 호출.
#
# 처리 흐름:
#   1. 결과 저장 디렉터리 생성
#      예: results/project/group/test/
#   2. get_dataloaders(seed, test) → 클래스별 DataLoader 리스트 생성
#   3. GPU 디바이스 설정
#   4. 클래스별 루프 (예: bracket_black, bracket_brown, ...):
#      a. 시드 고정 (재현성)
#      b. get_glass(imagesize, device) → GLASS 인스턴스 리스트 생성
#      c. GLASS.set_model_dir() → 모델/결과 저장 경로 설정
#      d. test='ckpt' → GLASS.trainer() 학습 실행
#         flag = (i_auroc, i_ap, p_auroc, p_ap, p_pro, best_epoch)
#         flag가 int이면 distribution 판별 결과 (Manifold=0, Hypersphere=1)
#      e. GLASS.tester() → 최적 체크포인트로 최종 평가
#      f. 결과 CSV 누적 저장
#   5. 분포 판별 결과를 xlsx로 저장
#      예: datasets/excel/mpdd_distribution.xlsx
#
# 결과 저장 구조:
#   results/
#   └── project/group/test/
#       ├── models/
#       │   └── backbone_0/
#       │       └── mpdd_bracket_black/
#       │           ├── ckpt_best.pth     ← 최적 에폭 체크포인트
#       │           ├── tb/               ← TensorBoard 로그
#       │           └── eval/             ← 추론 결과 시각화
#       │               ├── 000_pred.jpg  ← [원본|GT마스크|예측히트맵] 3열
#       │               └── ...
#       └── results.csv                   ← 전체 클래스 성능 지표
#
# 성능 지표 (result_collect):
#   image_auroc : 이미지 단위 이상탐지 AUROC (0~1, 높을수록 좋음)
#   image_ap    : 이미지 단위 Average Precision
#   pixel_auroc : 픽셀 단위 분할 AUROC
#   pixel_ap    : 픽셀 단위 Average Precision
#   pixel_pro   : Per-Region Overlap AUC (결함 영역 커버리지)
#   best_epoch  : 최고 성능 에폭 번호
# =============================================================================
@main.result_callback()
def run(
        methods,
        results_path,
        gpu,
        seed,
        log_group,
        log_project,
        run_name,
        test,
):
    """
    GLASS 학습/추론 메인 루프.

    Args:
        methods:      net()과 dataset()의 반환값 리스트
                      → dict 변환: {"get_glass": fn, "get_dataloaders": fn}
        results_path: 결과 저장 루트 경로 ("results")
        gpu:          GPU 번호 튜플 (0,) 또는 (0, 1)
        seed:         전역 랜덤 시드
        log_group:    결과 폴더 그룹명
        log_project:  결과 폴더 프로젝트명
        run_name:     실행 이름
        test:         'ckpt'=학습+평가, 기타=평가만
    """
    # ── (key, factory_fn) 튜플 리스트 → 딕셔너리 변환 ──────────────────────
    # net()    → ("get_glass", get_glass 함수)
    # dataset() → ("get_dataloaders", get_dataloaders 함수)
    # methods = {"get_glass": fn, "get_dataloaders": fn}
    methods = {key: item for (key, item) in methods}

    # ── 결과 저장 경로 생성 ──────────────────────────────────────────────────
    # utils.create_storage_folder: results/{log_project}/{log_group}/{run_name}/
    # mode="overwrite": 이미 존재하면 덮어씀
    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    # ── 클래스별 DataLoader 생성 ─────────────────────────────────────────────
    # get_dataloaders(seed=0, test='ckpt') 호출
    # 반환: [{"training": dl, "testing": dl}, ...]  (클래스 수만큼)
    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    # ── GPU 디바이스 설정 ────────────────────────────────────────────────────
    # utils.set_torch_device([0]) → torch.device("cuda:0")
    # 복수 GPU: [0, 1] → cuda:0 (DataParallel은 별도 설정 필요)
    device = utils.set_torch_device(gpu)

    # ── 결과 수집 및 분포 판별 결과 초기화 ──────────────────────────────────
    result_collect = []  # 클래스별 성능 지표 누적 리스트
    data = {'Class': [], 'Distribution': [], 'Foreground': []}
    df = pd.DataFrame(data)  # 분포 판별(svd) 결과 저장용 DataFrame

    # ── 클래스별 학습/평가 루프 ──────────────────────────────────────────────
    # 예: subdatasets=["bracket_black", "bracket_brown"]이면 2회 반복
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):

        # 클래스마다 동일한 시드로 고정 → 클래스 간 결과 재현성 보장
        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name       # 예: "mpdd_bracket_black"
        imagesize    = dataloaders["training"].dataset.imagesize  # 288

        # ── GLASS 인스턴스 생성 ──────────────────────────────────────────────
        # get_glass((288, 288), device) → [GLASS 인스턴스]
        # backbone 여러 개이면 인스턴스도 여러 개
        glass_list = methods["get_glass"](imagesize, device)

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )

        # ── 모델 저장 디렉터리 생성 ──────────────────────────────────────────
        # run_save_path/models/ → backbone별로 하위 폴더 분리
        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)

        for i, GLASS in enumerate(glass_list):
            # flag 초기값: (i_auroc, i_ap, p_auroc, p_ap, p_pro, best_epoch)
            # best_epoch=-1: 아직 학습/평가 미완
            flag = 0., 0., 0., 0., 0., -1.

            # backbone별 시드 고정 (backbone.seed가 있는 경우)
            if GLASS.backbone.seed is not None:
                utils.fix_seeds(GLASS.backbone.seed, device)

            # ── 모델 디렉터리 설정 ───────────────────────────────────────────
            # GLASS.set_model_dir(models_dir/backbone_0, mpdd_bracket_black)
            # → 체크포인트 저장/로드 경로: models/backbone_0/mpdd_bracket_black/ckpt_best.pth
            # → TensorBoard 경로: models/backbone_0/mpdd_bracket_black/tb/
            # → eval 시각화 경로: models/backbone_0/mpdd_bracket_black/eval/
            GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)

            if test == 'ckpt':
                # ── 학습 실행 ────────────────────────────────────────────────
                # GLASS.trainer(train_dataloader, test_dataloader, dataset_name)
                # 내부 처리:
                #   1. 분포 자동 판별 (svd=0이면 FFT 분석으로 Manifold/Hypersphere 결정)
                #   2. 분포 판별 결과가 int이면 flag=int로 반환 (→ xlsx 저장)
                #   3. meta_epochs번 학습:
                #      - center c 갱신 (정상 feature 평균)
                #      - _train_discriminator() × limit
                #        · GAS: Gaussian noise → Gradient Ascent → BCE
                #        · LAS: Perlin+DTD 합성 → FocalLoss
                #      - 매 eval_epochs마다 _evaluate()로 AUC 측정
                #      - 최고 AUC 에폭에서 ckpt_best.pth 저장
                #   4. flag = (i_auroc, i_ap, p_auroc, p_ap, p_pro, best_epoch) 반환
                flag = GLASS.trainer(dataloaders["training"], dataloaders["testing"], dataset_name)

                # flag가 int이면 distribution 판별만 수행한 것 (학습 미진행)
                # svd가 자동으로 분포를 판별하고 xlsx에 기록
                if type(flag) == int:
                    row_dist = {'Class': dataloaders["training"].name, 'Distribution': flag, 'Foreground': flag}
                    df = pd.concat([df, pd.DataFrame(row_dist, index=[0])])

            if type(flag) != int:
                # ── 최종 평가 ────────────────────────────────────────────────
                # GLASS.tester(test_dataloader, dataset_name)
                # ckpt_best.pth 로드 후 _predict() + _evaluate() 실행
                # 반환: (image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, best_epoch)
                i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = GLASS.tester(dataloaders["testing"], dataset_name)
                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "image_auroc":  i_auroc,   # 이미지 단위 AUROC
                        "image_ap":     i_ap,      # 이미지 단위 AP
                        "pixel_auroc":  p_auroc,   # 픽셀 단위 AUROC
                        "pixel_ap":     p_ap,      # 픽셀 단위 AP
                        "pixel_pro":    p_pro,     # Per-Region Overlap AUC
                        "best_epoch":   epoch,     # 최적 에폭 번호
                    }
                )

                if epoch > -1:
                    # 콘솔 출력: 소수점 2자리까지 백분율로 표시
                    # 예: "image_auroc:95.32 image_ap:91.45 ..."
                    for key, item in result_collect[-1].items():
                        if isinstance(item, str):
                            continue
                        elif isinstance(item, int):
                            print(f"{key}:{item}")
                        else:
                            print(f"{key}:{round(item * 100, 2)} ", end="")

                # ── 클래스별 결과 CSV 즉시 저장 ─────────────────────────────
                # 클래스 하나가 끝날 때마다 저장 → 중간 실패 시 손실 최소화
                # 저장 위치: run_save_path/results.csv
                print("\n")
                result_metric_names  = list(result_collect[-1].keys())[1:]      # dataset_name 제외
                result_dataset_names = [results["dataset_name"] for results in result_collect]
                result_scores        = [list(results.values())[1:] for results in result_collect]
                utils.compute_and_store_final_results(
                    run_save_path,
                    result_scores,
                    result_metric_names,
                    row_names=result_dataset_names,
                )

    # ── 분포 판별 결과 xlsx 저장 ─────────────────────────────────────────────
    # svd=0(자동판별)일 때 각 클래스의 Manifold/Hypersphere 판별 결과를 xlsx로 저장
    # 저장 위치: datasets/excel/{dataset}_distribution.xlsx
    # 예: datasets/excel/mpdd_distribution.xlsx
    # 다음 실행 시 이 파일을 참조하여 svd 모드를 자동 선택
    if len(df['Class']) != 0:
        os.makedirs('./datasets/excel', exist_ok=True)
        xlsx_path = './datasets/excel/' + dataset_name.split('_')[0] + '_distribution.xlsx'
        df.to_excel(xlsx_path, index=False)


if __name__ == "__main__":
    # ── 경고 억제 ────────────────────────────────────────────────────────────
    # UserWarning (HuggingFace PyTorch 버전 체크 등 무관한 경고) 억제
    warnings.filterwarnings('ignore')

    # ── 로깅 설정 ────────────────────────────────────────────────────────────
    # INFO 레벨: 학습 진행상황, 데이터셋 로드 정보 등 출력
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))

    # ── Click CLI 실행 ───────────────────────────────────────────────────────
    # sys.argv를 파싱하여 서브커맨드(net, dataset) 순서대로 실행
    # 최종적으로 result_callback(run)이 호출됨
    main()
