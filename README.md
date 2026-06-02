# Industrial Incident GraphRAG

Knowledge Graph 기반 산업 설비 사고 원인 분석 시스템입니다.  
자연어 쿼리를 입력하면 LLM Agent가 Neo4j 그래프를 탐색하며 사고 원인, 관련 컴포넌트, 대응 절차를 종합 분석합니다.

---

## 프로젝트 개요

산업 현장의 설비 고장은 단일 원인이 아닌 에러 코드 → 컴포넌트 → 과거 사고 → 정비 절차로 이어지는 복잡한 관계망을 가집니다.  
이 시스템은 해당 관계를 Knowledge Graph로 모델링하고, Hybrid Retrieval + Reflection Loop를 통해 근거 있는 분석을 생성합니다.

- **데이터**: AI4I 2020 공개 데이터셋 (10,000행) + 도메인 특화 생성 데이터
- **그래프**: 183 nodes · 513 edges · 9 node types · 17 edge types
- **핵심**: ErrorCode를 허브로 Incident / Component / ManualSection / Image / SensorGraph가 연결된 Heterogeneous Graph

---

## 주요 기능

### Hybrid Retrieval
Dense 벡터 유사도 + Knowledge Graph 신호를 가중합으로 결합합니다.

```
Manual   : 0.5 × vector_score + 0.2 × graph_score
Incident : 0.5 × vector_score + 0.3 × severity_score + 0.2 × graph_score
Image    : 0.7 × clip_score   + 0.3 × graph_score
```

graph_score는 쿼리에서 발견된 에러코드와 검색 결과의 에러코드 간 겹침 비율로 계산됩니다.

### Reflection Loop
계획된 탐색 스텝 소진 후 충분도(`sufficiency_threshold = 0.55`)를 판단해, 증거가 부족하면 LLM이 추가 탐색 스텝을 자체 생성하여 루프를 재진입합니다 (최대 5회).

### 환각 억제 (EvidenceState)
- `discovered_error_codes` 등 KG 실존 확인 ID만 누적 — LLM이 hallucinate한 코드는 상태에 추가되지 않음
- `_ARG_VALIDATORS` 정규식 강제: `E-\d{3}`, `C-\d{3}`, `INC-\d{4}`, `TWF|HDF|PWF|OSF|RNF`
- `already_visited()` 중복 툴 호출 차단

### 멀티모달 검색
CLIP(`open-clip-torch`) 텍스트 임베딩으로 센서 그래프 이미지 및 설비 사진을 검색합니다.

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Web Framework | FastAPI 0.136 + Uvicorn |
| Graph DB | Neo4j 6.2 (vector index 내장) |
| LLM | Claude (claude-sonnet-4-6) via Gateway |
| Embedding | OpenAI text-embedding-3-small |
| Multimodal | CLIP (open-clip-torch 3.3) |
| Image Analysis | Gemini 2.5 Pro |
| Data | Pandas, NumPy, Matplotlib |
| Validation | Pydantic v2 |

---

## 아키텍처

```
Client
  │
  ▼
FastAPI  (main.py)
  ├── /api/v1/search      → LLMService / RetrievalService
  ├── /api/v1/incidents   → RetrievalService
  └── /api/v1/agent       → AgentService
          │
          ▼
    AgentService
      ├── Planner          : LLM이 탐색 계획 생성 (JSON)
      ├── Explorer         : KG 툴 실행 + EvidenceState 누적
      │     ├── Reflection Loop (max 5회)
      │     └── Hybrid Retrieval (vector + graph)
      ├── Validator        : 증거 일관성 검증
      └── Synthesizer      : 최종 분석 보고서 생성
          │
          ▼
    Neo4j Graph DB
      ├── vector index  (manual_embedding, incident_embedding)
      └── image index   (image_clip_text_embedding)
```

### 디렉터리 구조

```
backend/
├── api/                  # FastAPI 라우터
│   ├── agent.py
│   ├── incidents.py
│   └── search.py
├── agent/                # LLM Agent 파이프라인
│   ├── explorer.py       # Reflection Loop, EvidenceState, ARG_VALIDATORS
│   ├── planner.py
│   ├── synthesizer.py
│   └── validator.py
├── graph/                # Neo4j 연동
│   ├── builder.py        # 노드/엣지 삽입 (DB 초기화)
│   ├── context_builder.py
│   └── queries.py
├── retrieval/
│   └── vector_retriever.py  # Hybrid Retrieval 핵심
├── services/
│   ├── agent_service.py
│   ├── llm_service.py
│   └── retrieval_service.py
├── multimodal/
│   ├── mappings.py       # ErrorCode ↔ Component/Incident/Manual 매핑 정의
│   └── image_analyzer.py
├── pipeline/             # 임베딩 파이프라인
│   ├── embed_manuals.py
│   ├── embed_incidents.py
│   └── embed_images_clip.py
├── scripts/              # 데이터 생성 스크립트
│   ├── generate_incidents_ai4i.py
│   ├── generate_processed_data.py
│   ├── generate_sensor_graphs.py
│   └── generate_dashboards.py
├── data/
│   ├── processed/        # JSON 노드 데이터
│   └── raw/images/       # 정적 이미지 파일
├── core/
│   └── config.py         # 환경변수 설정
└── main.py
```

---

## API

Base URL: `http://localhost:8000/api/v1`  
Swagger UI: `http://localhost:8000/docs`

### Agent

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/agent/investigate` | 쿼리 기반 사고 원인 분석 (Reflection Loop) |

```json
// Request
{
  "query": "베어링 과열 및 진동 이상 발생",
  "use_vector": true,
  "max_iterations": 5
}
```

### Search

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/search/query` | 자연어 질의 → GraphRAG 답변 |
| GET | `/search/context/error-code/{code}` | 에러코드 기반 컨텍스트 조회 |
| GET | `/search/context/failure-type/{type}` | 고장 유형 기반 컨텍스트 조회 |

### Incidents

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/incidents/{incident_id}/similar` | 유사 사고 조회 |
| GET | `/incidents/procedures/{error_code}` | 에러코드별 대응 절차 조회 |

### 정적 파일

| 경로 | 설명 |
|------|------|
| `GET /images/{category}/{image_id}.jpg` | 설비 이미지 |
| `GET /images/sensor_graphs/{id}.png` | 센서 그래프 |

---

## 세팅

### 1. 환경변수

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# OpenAI (임베딩)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# LLM Gateway (Claude)
GATEWAY_API_KEY=...
GATEWAY_BASE_URL=...
DEFAULT_LLM_MODEL=claude-sonnet-4-6

# 멀티모달 (Gemini)
MULTIMODAL_MODEL=gemini-2.5-pro
OPENROUTER_API_KEY=...
```

### 2. 패키지 설치

```bash
cd backend
pip install -r requirements.txt
```

### 3. 데이터 생성 및 그래프 초기화

```bash
# ai4i2020.csv를 프로젝트 루트에 위치시킨 후 실행

# 1. AI4I 기반 Incident / FailurePattern 생성
python -m scripts.generate_incidents_ai4i

# 2. 센서 그래프 이미지 생성
python -m scripts.generate_sensor_graphs

# 3. 대시보드 이미지 생성
python -m scripts.generate_dashboards

# 4. image_metadata 기반 보조 JSON 생성
python -m scripts.generate_processed_data

# 5. Neo4j 노드/엣지 삽입
python -m graph.builder

# 6. 텍스트 임베딩 (manual, incident)
python -m pipeline.embed_manuals
python -m pipeline.embed_incidents

# 7. CLIP 이미지 임베딩
python -m pipeline.embed_images_clip
```

### 4. 서버 실행

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 그래프 스키마 요약

```
FailurePattern ──[MAPS_TO]──────► ErrorCode ──[INVOLVES]──────► Component
                                      │                               ▲
                               [REFERENCES]                    [AFFECTS] [DEPICTS]
                                      │                               │
                                 ManualSection              Incident ─┤
                                      │                    (AI4I)    │
                               [HAS_PROCEDURE]                  [LINKED_IMAGE]
                                      │                          [LINKED_GRAPH]
                                  Procedure              Image / SensorGraph / Dashboard
                                                              │
                                                        [SHOWS_ERROR]
                                                              │
                                                          ErrorCode
```

노드 타입: `ErrorCode` `Component` `ManualSection` `Procedure` `Incident` `FailurePattern` `Image` `SensorGraph` `Dashboard`

---

## 추후 개선사항

### 배포 (진행 예정)

- **Docker Compose** 구성 — FastAPI + Neo4j + 임베딩 파이프라인을 컨테이너로 묶어 단일 명령으로 실행 가능하도록 정리 중
- **클라우드 배포** — Neo4j AuraDB(관리형) + 백엔드 서버 분리 배포 구조로 전환 예정
- **환경 분리** — `dev` / `prod` 설정 파일 분리, 시크릿 관리 도구 적용

### Retrieval 개선

- **BM25 + Dense 혼합** — 현재는 Dense 벡터 + KG 신호 조합이지만, BM25 키워드 검색을 추가해 Reciprocal Rank Fusion(RRF)으로 세 점수를 통합하는 방향 검토
- **Cross-encoder Reranker** — 현재 선형 가중합 방식 대신 학습된 reranker 모델을 최종 순위 결정 단계에 적용
- **쿼리 확장** — 동의어 및 도메인 용어 사전을 활용한 쿼리 rewriting

### 데이터 & 그래프

- **실 공장 데이터 연동** — AI4I 2020 샘플 외 실제 설비 로그 / SCADA 데이터 수집 파이프라인 구성
- **그래프 자동 업데이트** — 신규 사고 발생 시 Neo4j 노드/엣지를 자동으로 추가하는 이벤트 기반 파이프라인
- **시계열 센서 데이터** — 현재 이미지로 저장된 센서 그래프를 원시 시계열 값으로 보관하고, 이상 탐지 모델과 직접 연동

### Agent 개선

- **멀티턴 대화** — 현재 단일 쿼리 기반이지만, 이전 분석 결과를 맥락으로 이어받는 대화형 인터페이스 지원
- **Confidence 산출 고도화** — `_is_sufficient()` 의 단순 임계값 판단을 LLM 기반 자기 평가로 대체
- **툴 확장** — 실시간 센서 조회 / 부품 재고 시스템 / 작업 지시서 연동 툴 추가

### 관측성 & 운영

- **요청 로깅** — 쿼리별 툴 호출 횟수, Reflection 발생 여부, 응답 시간 추적
- **API 인증** — 현재 인증 없음 → API Key 또는 JWT 기반 인증 추가 필요
- **Rate limiting** — LLM API 비용 제어를 위한 요청 제한 적용
