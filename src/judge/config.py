# 3차원 평가 가중치
DIMENSION_WEIGHTS = {"argument": 3, "evidence": 2, "language": 2}
TOTAL_WEIGHT = 7
PERCENT_BASELINE = 5.0
PERCENT_STEP = 3.0

# 반박 단계 식별자
REBUTTAL_PHASES = {"chained_rebuttal", "free_rebuttal", "role_reversal", "synthesis"}

# 지수이동평균 평활 계수 (0~1, 높을수록 최근 발언 가중)
EMA_ALPHA = 0.4

# speech_memory 최근 유지 개수
RECENT_WINDOW = 5

# 모델 설정
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DRIVE_PATH = "/content/drive/MyDrive/models/Qwen2.5-14B-Instruct"
