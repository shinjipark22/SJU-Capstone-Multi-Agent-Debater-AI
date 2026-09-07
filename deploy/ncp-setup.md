# 네이버 클라우드 서버 만들기 (처음부터 끝까지)

크레딧 10만원 기준. 콘솔 메뉴 이름은 2026년 9월 신규 콘솔 기준이다.

## 0. 시작 전에 — 크레딧을 어디에 쓰는지

10만원은 생각보다 빨리 녹는다. 과금 항목이 서버 하나가 아니다.

| 항목 | 대략 |
| --- | --- |
| 서버 (2vCPU / 8GB, Standard) | 월 7만원 안팎 |
| 스토리지 50GB | 월 5천원 안팎 |
| 공인 IP 1개 | 월 4천원 안팎 |

정확한 금액은 콘솔의 **요금 계산기**(ncloud.com → 요금 → 요금 계산기)에서 지역·세대별로 확인할 것.
위 조합이면 **10만원으로 약 1개월** 돌릴 수 있다는 뜻이다. 그래서:

- **시간 요금제**를 고른다 (월 요금제는 안 쓸 때도 나간다).
- 실험 안 하는 시간에는 콘솔에서 서버를 **정지**한다. 정지 상태면 서버 요금은 멈추고
  스토리지·공인 IP 요금만 남는다.
- 실험이 끝나면 서버를 **반납**하고 공인 IP도 함께 **반납**한다. IP만 남겨두면 계속 과금된다.

## 1. 회원가입 · 결제수단 등록

1. [ncloud.com](https://www.ncloud.com) 회원가입 → 로그인
2. 우측 상단 **마이페이지 → 결제수단 관리**에서 카드 등록 (크레딧이 있어도 등록은 필요)
3. **마이페이지 → 크레딧**에서 보유 크레딧 확인

## 2. VPC · Subnet 만들기 (서버보다 먼저)

신규 콘솔은 VPC 환경이라 네트워크를 먼저 만들어야 서버를 만들 수 있다.

1. 콘솔 상단 **Services**(또는 전체 서비스) → **Networking → VPC**
2. **VPC 생성**
   - 이름: `debate-vpc`
   - IP 주소 범위: `10.0.0.0/16` (기본값 그대로)
3. 왼쪽 메뉴 **Subnet** → **Subnet 생성**
   - 이름: `debate-subnet`
   - VPC: 방금 만든 `debate-vpc`
   - IP 범위: `10.0.1.0/24`
   - 가용 영역: 아무거나 (예: KR-1)
   - Internet Gateway 전용 여부: **공인(Public)**  ← 외부에서 접속해야 하므로 필수
   - 용도: **일반(GENERAL)**

## 3. 서버 생성

**Services → Compute → Server → 서버 생성**

| 단계 | 선택 |
| --- | --- |
| 서버 이미지 | OS 이미지 → **Ubuntu 22.04** (또는 24.04) |
| 서버 타입 | **Standard**, **2vCPU / 8GB** (동시 참가자 5명 이상이면 4vCPU/16GB) |
| 요금제 | **시간 요금제** |
| 부팅 디스크 | 50GB (기본값이 더 크면 줄여도 된다) |
| VPC / Subnet | `debate-vpc` / `debate-subnet` |
| 공인 IP | **새로 할당** ← 안 하면 외부에서 접속 불가 |
| 서버 이름 | `debate-api` |

**인증키**: "새로운 인증키 생성" → 이름 입력 → **`.pem` 파일 다운로드**.
이 파일을 잃어버리면 서버에 못 들어간다. 백업해 둘 것.

**ACG(방화벽)**: 기본 ACG 를 그대로 두고 생성한 뒤 4번에서 규칙을 넣는다.

**서버 생성** 클릭 → 상태가 `운영중`이 될 때까지 5~10분 기다린다.

## 4. 방화벽(ACG) 규칙

**Services → Compute → Server → ACG** → 해당 ACG 선택 → **ACG 설정**

| 프로토콜 | 접근 소스 | 허용 포트 | 용도 |
| --- | --- | --- | --- |
| TCP | `내 IP/32` | 22 | SSH (내 IP 만) |
| TCP | `0.0.0.0/0` | 80 | HTTP |
| TCP | `0.0.0.0/0` | 443 | HTTPS |

**8001(FastAPI)과 8000(vLLM)은 절대 열지 않는다.** FastAPI 는 127.0.0.1 에만 붙이고 nginx 가
앞에서 받는다. vLLM 은 SSH 리버스 터널로만 연결한다.

## 5. 서버 접속

1. 서버 목록에서 서버 선택 → **서버 관리 및 설정 변경 → 관리자 비밀번호 확인**
   → 다운로드한 `.pem` 파일 내용을 붙여넣으면 `root` 비밀번호가 나온다.
2. 접속:

```bash
chmod 400 ~/Downloads/debate-key.pem
ssh root@<공인IP>          # 위에서 확인한 비밀번호 입력
```

3. 작업용 계정 만들기 (root 로 계속 쓰지 않는다):

```bash
adduser ubuntu && usermod -aG sudo ubuntu
mkdir -p /home/ubuntu/.ssh && cp ~/.ssh/authorized_keys /home/ubuntu/.ssh/ 2>/dev/null || true
chown -R ubuntu:ubuntu /home/ubuntu/.ssh
```

## 6. 앱 배포

여기부터는 [deploy/README.md](README.md) 의 2~4번을 그대로 따라가면 된다.
요약하면: 코드 clone → venv 설치 → `.env` 작성 → systemd 등록 → nginx 설정 → 프론트 `dist/` 업로드.

## 7. 연구실 GPU 연결

서버가 뜬 뒤, **연구실 머신에서** 실행한다 (서버에서 하는 게 아니다).

```bash
# 연구실 머신 → 서버로 키 등록 (한 번만)
ssh-copy-id ubuntu@<공인IP>

sudo apt install -y autossh
export NCP_HOST=ubuntu@<공인IP>
bash scripts/serve/start_gpu_tunnel.sh
```

이제 서버 안에서 `curl http://127.0.0.1:8000/v1/models` 가 응답하면 연결된 것이다.
서버의 `.env` 는 `VLLM_BASE_URL=http://127.0.0.1:8000/v1` 그대로 두면 된다.

터널이 항상 떠 있어야 하므로, 연구실 머신에서 `systemd --user` 나 `tmux` 로 상시 실행할 것.

## 8. 마무리 점검

```bash
curl https://<도메인 또는 공인IP>/health     # {"status":"ok",...}
curl https://<도메인 또는 공인IP>/topics     # 논제 목록 (터널이 없어도 응답)
```

토론을 한 판 돌려보고 `GET /sessions` 에 행이 쌓이는지까지 확인하면 끝이다.

## 실험 끝난 뒤

1. **데이터부터 내려받는다**: `scp ubuntu@<IP>:/opt/debate/data/sessions.db .`
   (또는 `/sessions/export.csv`, `/surveys/export.csv` 다운로드)
2. 서버 **반납** → 공인 IP **반납** → 스토리지 남아 있으면 함께 삭제
3. 크레딧 잔액 확인
