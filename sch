"""
MaintenanceScheduler: 예방 정비 스케줄링 의사결정 모듈 (특허 Step 3)

특허 청구항 대응:
  [Step 3-1] DualChannelEdgeGATConv의 zone_urgency를 입력으로 받아
             긴급도 임계값을 초과하는 구역 식별
  [Step 3-2] 물동량 예측(24슬롯 일별 패턴)과 긴급도를 결합해
             전체 처리량(Throughput) 손실이 최소인 정비 타임슬롯 결정
  [Step 3-3] 정비 리소스(작업자·OHT) 투입 우선순위 및 소요 기간 추정

독창성 포인트:
  - 단순 주기 점검이 아닌 urgency 기반 동적 스케줄링
  - Sliding-window 최소 유량 창 탐색으로 Throughput 저하 최소화
  - urgency 수준에 비례한 정비 기간 추정 (과잉 정비 방지)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

# 정규화된 24시간 물동량 패턴 (새벽↓ → 오전 피크↑ → 오후 유지 → 저녁↓)
HOURLY_PATTERN = [
    0.30, 0.20, 0.10, 0.10, 0.20, 0.50,   # 00-05
    0.80, 1.00, 0.90, 0.85, 0.80, 0.75,   # 06-11
    0.70, 0.75, 0.80, 0.85, 0.90, 1.00,   # 12-17
    0.80, 0.60, 0.50, 0.40, 0.35, 0.30,   # 18-23
]


@dataclass
class MaintenanceTask:
    zone_id:   int
    slot:      int    # 정비 시작 슬롯 (0-based, 0=00:00)
    duration:  int    # 소요 슬롯 수
    urgency:   float  # 긴급도 스코어 (0~1)
    flow_cost: float  # 정비 창 내 총 물동량 (Throughput 손실 지표)


class MaintenanceScheduler:
    """
    존별 긴급도(Urgency Score)를 바탕으로 최적 정비 타임슬롯을 결정한다.

    알고리즘:
      1. urgency > threshold 인 존 식별 (내림차순 정렬 → 리소스 우선순위)
      2. 존별 24슬롯 물동량 예측
      3. Sliding-window로 최소 유량 창 탐색
      4. urgency 수준에 따른 정비 소요 기간 추정

    Parameters
    ----------
    n_slots           : 하루 시간 슬롯 수 (기본 24 = 시간 단위)
    urgency_threshold : 정비 트리거 임계값
    min_duration      : 최소 정비 소요 슬롯
    max_duration      : 최대 정비 소요 슬롯
    """

    def __init__(
        self,
        n_slots: int           = 24,
        urgency_threshold: float = 0.5,
        min_duration: int      = 1,
        max_duration: int      = 6,
    ):
        self.n_slots           = n_slots
        self.urgency_threshold = urgency_threshold
        self.min_duration      = min_duration
        self.max_duration      = max_duration

    # ── 핵심 스케줄링 ─────────────────────────────────────────────────────────

    def schedule(
        self,
        zone_urgency: torch.Tensor,       # (Z,)
        zone_flow_forecast: torch.Tensor, # (Z, T)
    ) -> Dict[int, MaintenanceTask]:
        """
        긴급도 임계값을 초과하는 모든 존에 대해 최적 정비 타임슬롯을 반환.
        결과는 urgency 내림차순으로 정렬 (정비 리소스 투입 우선순위).
        """
        urgency_list  = zone_urgency.detach().cpu().tolist()       # List[float]
        forecast_list = zone_flow_forecast.detach().cpu().tolist() # List[List[float]]

        urgent_zones = sorted(
            [z for z in range(len(urgency_list)) if urgency_list[z] > self.urgency_threshold],
            key=lambda z: urgency_list[z],
            reverse=True,
        )

        tasks: Dict[int, MaintenanceTask] = {}
        for z in urgent_zones:
            u        = float(urgency_list[z])
            duration = self._estimate_duration(u)
            slot, flow_cost = self._find_low_traffic_window(forecast_list[z], duration)
            tasks[z] = MaintenanceTask(
                zone_id=z,
                slot=slot,
                duration=duration,
                urgency=round(u, 4),
                flow_cost=round(float(flow_cost), 4),
            )

        return tasks

    def forecast_traffic(self, zone_flow: torch.Tensor) -> torch.Tensor:
        """
        존별 일평균 유량 × 24시간 패턴 → (Z, 24) 예측 텐서.

        실제 시스템에서는 LSTM/Prophet 모델로 교체 가능.
        """
        flow_list = zone_flow.detach().cpu().tolist()   # List[float]
        forecast  = [
            [f * float(HOURLY_PATTERN[t]) for t in range(len(HOURLY_PATTERN))]
            for f in flow_list
        ]
        return torch.tensor(forecast, dtype=torch.float32)

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _estimate_duration(self, urgency: float) -> int:
        """
        긴급도 [threshold, 1] → 정비 소요 기간 [min, max] 선형 매핑.
        긴급도가 높을수록 더 많은 작업이 필요하다고 가정.
        """
        t = (urgency - self.urgency_threshold) / (1.0 - self.urgency_threshold + 1e-8)
        duration = self.min_duration + int(t * (self.max_duration - self.min_duration))
        return max(self.min_duration, min(self.max_duration, duration))

    def _find_low_traffic_window(
        self,
        flow: list,   # List[float], length T
        duration: int,
    ) -> tuple[int, float]:
        """
        Sliding-window로 total flow가 최소인 시작 슬롯을 반환.
        Returns (best_start_slot, total_flow_in_window)
        """
        T = len(flow)
        if duration >= T:
            return 0, sum(flow)

        min_cost  = float("inf")
        best_slot = 0
        for t in range(T - duration + 1):
            cost = sum(flow[t : t + duration])
            if cost < min_cost:
                min_cost  = cost
                best_slot = t
        return best_slot, min_cost

    # ── 출력 ──────────────────────────────────────────────────────────────────

    def print_schedule(self, tasks: Dict[int, MaintenanceTask]) -> None:
        if not tasks:
            print("  스케줄 없음: 긴급 정비 필요 구역 없음")
            return

        print(f"\n{'Zone':>5}  {'Urgency':>7}  {'Start':>5}  {'Duration':>8}  {'FlowCost':>9}")
        print("─" * 48)
        for task in sorted(tasks.values(), key=lambda t: t.urgency, reverse=True):
            h_start = task.slot % 24
            print(
                f"  {task.zone_id:3d}    {task.urgency:.3f}   "
                f"{h_start:02d}:00   +{task.duration}슬롯    {task.flow_cost:.4f}"
            )
