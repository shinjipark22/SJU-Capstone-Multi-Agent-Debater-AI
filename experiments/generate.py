"""
generate.py -- 모델별 48개 토론 로그 생성 래퍼

각 모델에 대해 12 topics x 2 stances x 2 formats = 48개 실험을 실행한다.
서브프로세스로 _run_single.py를 호출하여 모델 전환 시 환경변수 격리를 보장한다.

사용법:
    python -m experiments.generate --model Qwen-2.5-32B-Instruct
    python -m experiments.generate --model GPT-5.4
    python -m experiments.generate --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from experiments.config import (
    API_COST_CONFIRM,
    API_MAX_EXPERIMENTS,
    HF_CACHE_DIR,
    INTENSITY_PRESETS,
    LOGS_DIR,
    MODELS,
    ModelConfig,
    PROJECT_ROOT,
    SINGLE_EXPERIMENT_TIMEOUT,
    TOPIC_IDS,
    VLLM_HEALTH_POLL_INTERVAL,
    VLLM_STARTUP_TIMEOUT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "experiments" / "generate.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── vLLM 서버 관리 ──────────────────────────────────────────────────────────

def start_vllm(config: ModelConfig) -> subprocess.Popen:
    """GPU 2,3에서 vLLM 서버를 시작한다."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.gpu_devices
    env["HF_HOME"] = HF_CACHE_DIR

    cmd = [
        "vllm", "serve", config.model_name,
        "--port", str(config.port),
        "--tensor-parallel-size", str(config.tensor_parallel),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(config.max_model_len),
        "--dtype", "float16",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--download-dir", HF_CACHE_DIR,
    ]
    if config.quantization:
        cmd.extend(["--quantization", config.quantization])

    logger.info("vLLM 시작: %s (GPU %s, port %d)", config.model_name, config.gpu_devices, config.port)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_for_vllm(base_url: str, timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    """vLLM 헬스체크가 성공할 때까지 대기한다."""
    import urllib.request
    import urllib.error

    health_url = f"{base_url.rstrip('/').replace('/v1', '')}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5):
                logger.info("vLLM 준비 완료: %s", base_url)
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(VLLM_HEALTH_POLL_INTERVAL)
    logger.error("vLLM 시작 타임아웃 (%d초): %s", timeout, base_url)
    return False


def stop_vllm(proc: Optional[subprocess.Popen]):
    """vLLM 서버를 종료한다."""
    if proc is None:
        return
    logger.info("vLLM 종료 중...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("vLLM 종료 완료")


# ── 실험 실행 ───────────────────────────────────────────────────────────────

def _build_output_path(model_id: str, topic_id: str, preset_key: str) -> Path:
    """실험 결과 JSON 경로를 생성한다."""
    return LOGS_DIR / model_id / f"{topic_id}_{preset_key}.json"


def run_single_experiment(
    config: ModelConfig,
    topic_id: str,
    preset_key: str,
) -> bool:
    """서브프로세스로 단일 실험을 실행한다."""
    output_path = _build_output_path(config.model_id, topic_id, preset_key)
    preset = INTENSITY_PRESETS[preset_key]
    fmt = preset["format"]

    # 이미 완료된 실험은 스킵
    if output_path.exists():
        try:
            with output_path.open() as f:
                data = json.load(f)
            if data.get("is_finished"):
                logger.info("스킵 (이미 완료): %s", output_path.name)
                return True
        except (json.JSONDecodeError, KeyError):
            pass

    env = os.environ.copy()
    env["VLLM_BASE_URL"] = config.base_url
    env["LLM_MODEL"] = config.model_name

    # 사용자 대행(GPT-4o-mini)용 OpenAI 키는 항상 전달
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key

    if config.model_type == "openai":
        if not openai_key:
            logger.error("OPENAI_API_KEY가 설정되지 않았습니다.")
            return False
        env["LLM_API_KEY"] = openai_key
    else:
        env["LLM_API_KEY"] = "fake"

    intensities_str = ",".join(str(i) for i in preset["intensities"])
    cmd = [
        sys.executable, "-m", "experiments._run_single",
        "--topic", topic_id,
        "--format", fmt,
        "--intensities", intensities_str,
        "--preset", preset_key,
        "--output", str(output_path),
    ]

    logger.info("실험 시작: %s / %s / %s (%s)", config.model_id, topic_id, preset_key, preset["label"])

    try:
        result = subprocess.run(
            cmd, env=env, cwd=str(PROJECT_ROOT),
            timeout=SINGLE_EXPERIMENT_TIMEOUT,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("실험 실패: %s\nSTDERR: %s", output_path.name, result.stderr[-500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("실험 타임아웃 (%d초): %s", SINGLE_EXPERIMENT_TIMEOUT, output_path.name)
        return False
    except Exception as e:
        logger.error("실험 오류: %s — %s", output_path.name, e)
        return False


def generate_for_model(model_id: str) -> dict:
    """특정 모델에 대해 48개 실험을 모두 실행한다."""
    if model_id not in MODELS:
        logger.error("알 수 없는 모델: %s", model_id)
        return {"model": model_id, "success": 0, "fail": 0, "skip": 0}

    config = MODELS[model_id]
    vllm_proc = None

    needs_vllm = (config.model_type == "vllm")

    try:
        # API 모델 비용 안전장치
        if config.model_type == "openai" and API_COST_CONFIRM:
            presets = list(INTENSITY_PRESETS.keys())
            total = len(TOPIC_IDS) * len(presets)
            est_cost = total * 35 * 3000 / 1_000_000 * 12  # 대략 추정 ($)
            print(f"\n{'='*60}")
            print(f"  ⚠️  API 모델 실험: {model_id}")
            print(f"  실험 수: {total}개")
            print(f"  예상 API 호출: ~{total * 35}회")
            print(f"  예상 비용: ~${est_cost:.0f}")
            print(f"{'='*60}")
            answer = input("  계속하시겠습니까? (y/N): ").strip().lower()
            if answer != "y":
                logger.info("사용자가 취소함")
                return {"model": model_id, "success": 0, "fail": 0, "cancelled": True}

        if needs_vllm:
            vllm_proc = start_vllm(config)
            if not wait_for_vllm(config.base_url):
                return {"model": model_id, "success": 0, "fail": 48, "skip": 0}

        success, fail = 0, 0
        presets = list(INTENSITY_PRESETS.keys())
        total = len(TOPIC_IDS) * len(presets)

        for i, topic_id in enumerate(TOPIC_IDS):
            for preset_key in presets:
                idx = i * len(presets) + presets.index(preset_key) + 1

                # API 모델 실험 수 제한
                if config.model_type == "openai" and API_MAX_EXPERIMENTS > 0:
                    if (success + fail) >= API_MAX_EXPERIMENTS:
                        logger.warning("API 실험 수 제한 도달 (%d개), 중단", API_MAX_EXPERIMENTS)
                        return {"model": model_id, "success": success, "fail": fail}

                logger.info("[%d/%d] %s — %s / %s", idx, total, model_id, topic_id, preset_key)

                ok = run_single_experiment(config, topic_id, preset_key)
                if ok:
                    success += 1
                else:
                    fail += 1

        return {"model": model_id, "success": success, "fail": fail}

    finally:
        if needs_vllm:
            stop_vllm(vllm_proc)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="모델별 토론 로그 생성")
    parser.add_argument(
        "--model", required=True,
        help="모델 ID (config.py의 MODELS 키) 또는 'all'",
    )
    args = parser.parse_args()

    if args.model == "all":
        for model_id in MODELS:
            result = generate_for_model(model_id)
            logger.info("모델 완료: %s", result)
    else:
        result = generate_for_model(args.model)
        logger.info("모델 완료: %s", result)


if __name__ == "__main__":
    main()
