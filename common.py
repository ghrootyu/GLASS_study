# =============================================================================
# GLASS 공통 모듈
#
# 전체 feature 추출 파이프라인:
#
#   이미지 (B, 3, 288, 288)
#       ↓ NetworkFeatureAggregator  (backbone hook으로 중간 feature 추출)
#   layer2 feature: (B, 512, 36, 36)
#   layer3 feature: (B, 1024, 18, 18)
#       ↓ PatchMaker.patchify (model.py)
#   layer2 patches: (B×1296, 512, 3, 3)
#   layer3 patches: (B×1296, 1024, 3, 3)  ← 36×36으로 bilinear 보간 후
#       ↓ Preprocessing (MeanMapper per layer)
#   merged: (B×1296, 2, 1536)  ← layer별로 512→1536, 1024→1536 압축
#       ↓ Aggregator (layer 평균)
#   final: (B×1296, 1536)      ← Discriminator 입력
# =============================================================================

import copy
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class Preprocessing(torch.nn.Module):
    """
    각 레이어의 patch feature를 동일한 차원(pretrain_embed_dimension)으로 압축.

    왜 필요한가?
        layer2 feature 채널=512, layer3 feature 채널=1024로 서로 다름.
        두 레이어를 합치기 위해 동일 차원(1536)으로 맞춤.

    구조:
        layer별로 MeanMapper를 하나씩 생성
        → 각 레이어 feature를 독립적으로 1536 차원으로 압축

    입력: [(B×1296, 512, 3, 3), (B×1296, 1024, 3, 3)]  ← 레이어별 patch features
    출력: (B×1296, 2, 1536)
          2 = 레이어 수, 1536 = pretrain_embed_dimension

    예시:
        layer2 patch: (20736, 512, 3, 3) → MeanMapper → (20736, 1536)
        layer3 patch: (20736, 1024, 3, 3) → MeanMapper → (20736, 1536)
        stack → (20736, 2, 1536)
    """

    def __init__(self, input_dims, output_dim):
        """
        Args:
            input_dims: 각 레이어의 채널 수 리스트 e.g. [512, 1024]
            output_dim: 압축 목표 차원 e.g. 1536 (pretrain_embed_dimension)
        """
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        # 레이어별로 독립적인 MeanMapper 생성
        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        """
        Args:
            features: [(B×1296, 512, 3, 3), (B×1296, 1024, 3, 3)]

        반환: (B×1296, 2, 1536)
        """
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))   # 각 레이어 → (B×1296, 1536)
        return torch.stack(_features, dim=1)    # → (B×1296, 2, 1536)


class MeanMapper(torch.nn.Module):
    """
    Patch feature를 목표 차원으로 압축 (adaptive average pooling).

    원리:
        patch feature를 1D로 펼친 후 adaptive_avg_pool1d로 원하는 크기로 압축.
        단순 평균으로 차원을 맞추므로 파라미터가 없음 (학습 불필요).

    예시 (layer2, preprocessing_dim=1536):
        입력: (20736, 512, 3, 3)
        reshape: (20736, 1, 512×3×3) = (20736, 1, 4608)
        avg_pool: (20736, 1, 1536)   ← 4608 → 1536으로 풀링
        squeeze: (20736, 1536)
    """

    def __init__(self, preprocessing_dim):
        """
        Args:
            preprocessing_dim: 목표 차원 e.g. 1536
        """
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        """
        Args:
            features: (N, C, k, k) e.g. (20736, 512, 3, 3)
        반환: (N, preprocessing_dim) e.g. (20736, 1536)
        """
        features = features.reshape(len(features), 1, -1)   # (N, 1, C×k×k)
        return F.adaptive_avg_pool1d(
            features, self.preprocessing_dim
        ).squeeze(1)                                          # (N, 1536)


class Aggregator(torch.nn.Module):
    """
    레이어별 feature를 하나로 합산 (target_embed_dimension으로 압축).

    Preprocessing 이후 (B×1296, 2, 1536)를 받아
    2개 레이어를 평균 → (B×1296, 1536)로 최종 압축.

    왜 단순 평균인가?
        layer2는 저수준(텍스처/엣지), layer3는 고수준(형태/의미) 정보.
        두 레이어를 균등하게 합산하여 두 수준의 특징을 모두 활용.

    입력: (B×1296, 2, 1536)
    출력: (B×1296, 1536)   ← Discriminator 입력

    예시:
        입력:  (20736, 2, 1536)  ← 2개 레이어 feature
        pool:  (20736, 1, 1536)  ← 평균
        reshape: (20736, 1536)   ← Discriminator에 입력
    """

    def __init__(self, target_dim):
        """
        Args:
            target_dim: 최종 feature 차원 e.g. 1536 (target_embed_dimension)
        """
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """
        Args:
            features: (N, num_layers, dim) e.g. (20736, 2, 1536)
        반환: (N, target_dim) e.g. (20736, 1536)
        """
        features = features.reshape(len(features), 1, -1)         # (N, 1, 2×1536)
        features = F.adaptive_avg_pool1d(features, self.target_dim)# (N, 1, 1536)
        return features.reshape(len(features), -1)                 # (N, 1536)


class RescaleSegmentor:
    """
    Patch score 격자를 원본 이미지 크기로 upsample하여 segmentation map 생성.

    학습 후 추론 시 사용:
        patch_scores (B, 36, 36) → bilinear upsample → (B, 288, 288) → gaussian smooth

    smoothing=4 이유:
        36×36 격자를 288×288로 단순 확대하면 블록 아티팩트 발생.
        sigma=4의 가우시안 필터로 자연스럽게 경계를 부드럽게 처리.
    """

    def __init__(self, device, target_size=288):
        """
        Args:
            device:      연산 디바이스
            target_size: 출력 크기 (원본 이미지 크기와 동일하게 설정)
                         288 또는 (288, 288)
        """
        self.device      = device
        self.target_size = target_size
        self.smoothing   = 4  # gaussian filter sigma

    def convert_to_segmentation(self, patch_scores):
        """
        Patch score 격자를 원본 크기 segmentation map으로 변환.

        Args:
            patch_scores: (B, 36, 36) 각 patch의 이상 score

        반환: list of (288, 288) numpy array, 길이=B
              각 원소가 이미지 1장의 픽셀별 이상 score

        처리 과정:
            1. (B, 36, 36) → (B, 1, 36, 36) unsqueeze
            2. bilinear interpolate → (B, 1, 288, 288)
            3. squeeze → (B, 288, 288) numpy
            4. gaussian_filter(sigma=4) → 경계 부드럽게 처리

        예시:
            입력:  patch_scores[0] = [[0.1, 0.8, 0.1], ...]  (36×36)
            출력:  segmentation[0] = 288×288 부드러운 히트맵
                   높은 값(빨강)=이상, 낮은 값(파랑)=정상
        """
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)                 # (B, 1, 36, 36)
            _scores = F.interpolate(
                _scores,
                size=self.target_size,                     # 288
                mode="bilinear",
                align_corners=False,
            )                                              # (B, 1, 288, 288)
            _scores = _scores.squeeze(1)                   # (B, 288, 288)
            patch_scores = _scores.cpu().numpy()

        # gaussian smoothing: 블록 아티팩트 제거 및 경계 부드럽게
        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """
    Backbone에서 중간 레이어 feature를 추출하는 모듈.

    원리:
        PyTorch forward hook을 사용하여 backbone 순전파 중
        지정된 레이어의 출력을 가로챔.

        hook이 마지막 추출 레이어에 도달하면 예외를 발생시켜
        불필요한 이후 레이어 계산을 건너뜀 (효율화).

    MPDD 설정:
        layers_to_extract_from = ["layer2", "layer3"]
        layer2: (B, 512,  36, 36)  ← stride 8  (288/8=36)
        layer3: (B, 1024, 18, 18)  ← stride 16 (288/16=18)
    """

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        """
        Args:
            backbone:                사전학습된 backbone (WideResNet50)
            layers_to_extract_from: 추출할 레이어 이름 리스트 ["layer2", "layer3"]
            device:                  연산 디바이스
            train_backbone:          backbone 파라미터 학습 여부 (기본 False)
        """
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone               = backbone
        self.device                 = device
        self.train_backbone         = train_backbone

        # hook 핸들 초기화 (중복 등록 방지)
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        self.outputs = {}  # 레이어명 → feature 저장소

        # 지정된 각 레이어에 forward hook 등록
        for extract_layer in layers_to_extract_from:
            self.register_hook(extract_layer)

        self.to(self.device)

    def forward(self, images, eval=True):
        """
        이미지를 backbone에 통과시켜 중간 feature를 수집.

        Args:
            images: (B, 3, 288, 288)
            eval:   True이면 no_grad로 실행 (학습 시에도 backbone은 frozen)

        반환: {"layer2": (B, 512, 36, 36), "layer3": (B, 1024, 18, 18)}

        동작:
            backbone 순전파 중 hook이 각 레이어 출력을 self.outputs에 저장.
            마지막 레이어(layer3) 도달 시 예외를 던져 순전파 조기 종료.
        """
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass  # 마지막 추출 레이어 이후 계산 불필요 → 정상 종료
        return self.outputs

    def feature_dimensions(self, input_shape):
        """
        각 레이어의 feature 채널 수를 반환 (Preprocessing 초기화에 사용).

        Args:
            input_shape: (3, 288, 288)
        반환: [512, 1024]  ← layer2, layer3 채널 수
        """
        _input  = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

    def register_hook(self, layer_name):
        """지정된 레이어에 forward hook 등록."""
        module = self.find_module(self.backbone, layer_name)
        if module is not None:
            forward_hook = ForwardHook(
                self.outputs,
                layer_name,
                self.layers_to_extract_from[-1],  # 마지막 레이어 = "layer3"
            )
            if isinstance(module, torch.nn.Sequential):
                hook = module[-1].register_forward_hook(forward_hook)
            else:
                hook = module.register_forward_hook(forward_hook)
            self.backbone.hook_handles.append(hook)
        else:
            raise ValueError(f"Module {layer_name} not found in the model")

    def find_module(self, model, module_name):
        """backbone에서 이름으로 레이어 모듈 탐색."""
        for name, module in model.named_modules():
            if name == module_name:
                return module
            elif '.' in module_name:
                father, child = module_name.split('.', 1)
                if name == father:
                    return self.find_module(module, child)
        return None


class ForwardHook:
    """
    특정 레이어의 출력을 가로채는 forward hook.

    PyTorch hook 메커니즘:
        register_forward_hook(hook_fn) 등록 시
        해당 레이어의 forward 완료 후 hook_fn(module, input, output) 자동 호출.

    동작:
        - 레이어 출력(output)을 hook_dict에 저장
        - 마지막 추출 레이어이면 예외를 던져 이후 계산 건너뜀
    """

    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        """
        Args:
            hook_dict:             feature 저장 딕셔너리 (self.outputs)
            layer_name:            현재 hook이 달린 레이어 이름
            last_layer_to_extract: 마지막으로 추출할 레이어 이름 ("layer3")
        """
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        # 마지막 레이어면 True → 예외 발생으로 순전파 조기 종료
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        """
        hook 실행: 레이어 출력 저장 후 마지막 레이어면 예외 발생.

        예시:
            layer2 hook 실행 → hook_dict["layer2"] = (B, 512, 36, 36) 저장
            layer3 hook 실행 → hook_dict["layer3"] = (B, 1024, 18, 18) 저장
                             → LastLayerToExtractReachedException 발생
                             → backbone forward 중단 (layer4 계산 생략)
        """
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    """마지막 추출 레이어 도달 시 backbone 순전파를 중단하기 위한 예외."""
    pass
