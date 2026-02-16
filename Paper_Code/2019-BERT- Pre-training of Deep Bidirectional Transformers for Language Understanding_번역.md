# BERT: 언어 이해를 위한 깊은 양방향 트랜스포머의 사전 훈련 
### 원재 : 
~~~
2019 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova
Google AI Language
{jacobdevlin,mingweichang,kentonl,kristout}@google.com
~~~
## 원문과 번역, 어휘 설명

---

## 읽기 전에: 핵심 개념

이 문서에서 자주 나오는 **사전 훈련**과 **양방향 사전 훈련**을 먼저 정리한다.

### 사전 훈련 (pre-training)

**사전 훈련**이란, "감정 분석이야", "질문에 답해 줘" 같은 **목표 작업**에 모델을 바로 넣기 전에, 그 작업과는 별개로 **대량의 텍스트**만 가지고 먼저 학습해 두는 단계를 말한다. 이때 레이블(정답 표시)이 없어도 된다. 그냥 글만 많이 넣어 주고, "다음 단어 맞추기", "빈칸 채우기" 같은 간단한 목표로 학습시키면 된다.

이렇게 해 두면, 나중에 실제로 하고 싶은 작업(질의응답, 분류 등)에 맞출 때 **적은 데이터**만 있어도 잘 맞는다. "기초를 미리 쌓아 두는 것"이라고 보면 된다. **사전 훈련된** 모델은 "이미 그렇게 한 번 학습을 마친" 모델을 가리킨다.

### 양방향 사전 훈련 (bidirectional pre-training)

문장을 읽을 때 **왼쪽에서 오른쪽으로만** 보면서 학습하는 방식을 **단방향**이라고 한다. 반대로 **양방향**은, 어떤 단어를 이해할 때 **그 단어 앞쪽(왼쪽) 문맥과 뒤쪽(오른쪽) 문맥을 둘 다** 사용해서 학습하는 방식이다.

예를 들어 "그 동물은 [ ]을 먹었다"에서 빈칸을 채울 때, 단방향이면 "그", "동물은"까지만 보고 맞추지만, 양방향이면 "먹었다" 같은 뒤쪽 정보까지 함께 쓸 수 있다. 질의응답이나 개체명 인식처럼 **앞뒤 문맥이 다 필요한** 작업에는 양방향이 유리하다. BERT는 **마스크드 언어 모델(MLM)**, 즉 빈칸 채우기 방식으로 이 양방향을 **한 번에, 깊게** 사전 훈련한다. (예전 방식처럼 "왼쪽→오른쪽"과 "오른쪽→왼쪽"을 따로 훈련한 뒤 얕게 이어 붙이는 것과 다르다.)

---

## Abstract (초록)

### 원문 (1)
**We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.**

### 번역
이 논문에서는 **BERT**(Bidirectional Encoder Representations from Transformers)라는 새로운 언어 표현 모델을 제안한다. 기존 모델[^29]과 달리 BERT는 **레이블 없는 텍스트(unlabeled text)** 만으로 학습한다. 모든 층에서 왼쪽 문맥과 오른쪽 문맥을 **함께 사용해(jointly conditioning)** 깊은 양방향 표현을 만든다. 그래서 사전 훈련된 BERT는 질의응답, 언어 추론 같은 여러 작업에 **출력층 하나만** 붙여서 미세 조정(fine-tuning)하면 된다. 작업마다 구조를 크게 바꿀 필요 없이 최신 수준 성능을 낼 수 있다.

[^29]: Peters et al., 2018a; Radford et al., 2018

### 원문 (2)
**BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).**

### 번역
BERT는 구조는 단순한데, 실험 결과는 매우 좋다. 11개 자연어 처리 작업에서 당시 최고 성능을 갱신했다. 수치만 보면 GLUE 80.5%(7.7%p 상승), MultiNLI 86.7%(4.6%p 상승), SQuAD v1.1 Test F1 93.2(1.5pt 상승), SQuAD v2.0 Test F1 83.1(5.1pt 상승)이다.

#### 위 수치의 계산 방식 (계산식)

**1. GLUE 80.5%**

GLUE는 9개 작업(MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE, WNLI)의 점수를 하나로 묶은 값이다. 작업 \(t\)마다 사용하는 지표가 다르다(정확도, F1, 피어슨 상관 등). 각 작업의 원시 점수를 0~100 스케일로 정규화한 \(\text{score}_t\)를 쓰고, 이들의 평균이 GLUE 점수다.

\[
\text{GLUE} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \text{score}_t
\]

\(\mathcal{T}\): GLUE에 포함된 작업 집합(9개 또는 WNLI 제외 시 8개). \(\text{score}_t\): 해당 작업의 공식 지표를 0~100으로 환산한 값. 80.5는 이 평균값이다.

**2. MultiNLI 86.7% (정확도)**

MultiNLI는 문장 쌍에 대해 함의/모순/중립 3-class 분류 작업이다. 정확도는 맞힌 샘플 비율이다.

\[
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i = \hat{y}_i] = \frac{\text{맞힌 개수}}{N}
\]

\(N\): 전체 샘플 수. \(y_i\): \(i\)번째 정답 레이블. \(\hat{y}_i\): 예측 레이블. \(\mathbb{1}[\cdot]\): 조건 참이면 1, 거짓이면 0. 86.7%는 위 식으로 계산한 Test/Dev 정확도이다.


**3. SQuAD v1.1 / v2.0 Test F1**

질의응답에서 모델이 예측한 **답 구간(span)**과 정답 구간을 토큰 단위로 비교한다. 예측·정답 각각에 대해 겹치는 토큰 수로 **정밀도**(예측 구간 중 맞은 비율)·**재현율**(정답 구간 중 맞힌 비율)을 구하고, 질문 \(q\)마다 예측 구간 \(\text{pred}_q\), 정답 구간 \(\text{gold}_q\)(토큰 집합)에 대해 정밀도 \(P_q = |\text{pred}_q \cap \text{gold}_q| / |\text{pred}_q|\), 재현율 \(R_q = |\text{pred}_q \cap \text{gold}_q| / |\text{gold}_q|\), \(F1_q = 2 P_q R_q / (P_q + R_q)\)를 계산한 뒤, 테스트 F1 = \((1/Q) \sum_q F1_q\)(\(Q\): 질문 수)이다. 빈 구간이면 \(P_q = R_q = 0\). v2.0은 "답 없음"도 동일 방식으로 포함한다.

질문 \(q\)마다 위와 같이 \(P_q, R_q, F1_q\)를 구한 뒤, 테스트 F1은 \(F1_q\)의 평균이다. 수식:
\[
P_q = \frac{|\text{pred}_q \cap \text{gold}_q|}{|\text{pred}_q|}, \quad R_q = \frac{|\text{pred}_q \cap \text{gold}_q|}{|\text{gold}_q|}, \quad F1_q = \frac{2 P_q R_q}{P_q + R_q} = \frac{2 \, |\text{pred}_q \cap \text{gold}_q|}{|\text{pred}_q| + |\text{gold}_q|}
\]
\[
\text{F1} = \frac{1}{Q} \sum_{q=1}^{Q} F1_q
\]

93.2(v1.1), 83.1(v2.0)은 위 식으로 Test 세트에서 계산한 F1 평균값이다.

---

## Introduction (서론)

### 1문단

#### 원문
**Language model pre-training has been shown to be effective for improving many natural language processing tasks (Dai and Le, 2015; Peters et al., 2018a; Radford et al., 2018; Howard and Ruder, 2018). These include sentence-level tasks such as natural language inference (Bowman et al., 2015; Williams et al., 2018) and paraphrasing (Dolan and Brockett, 2005), which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained output at the token level (Tjong Kim Sang and De Meulder, 2003; Rajpurkar et al., 2016).**

#### 번역
**언어 모델 사전 훈련**(language model pre-training)이 여러 자연어 처리 작업 성능을 끌어올리는 데 도움이 된다는 건 이미 많이 알려져 있다[1–4]. 여기서 말하는 작업은 크게 두 종류다. 첫째, **문장 수준 작업(sentence-level tasks)** 다. 자연어 추론[5,6], 패러프레이즈[7]처럼 문장 전체를 보고 문장끼리 어떤 관계인지 예측하는 것이다. 둘째, **토큰 수준 작업(token-level tasks)**다. 개체명 인식[8], 질의응답[9]처럼 각 토큰(단어 단위)마다 레이블을 달아야 하는 작업을 말한다.

#### 각주 (부가 설명)
- [1] **Dai and Le (2015)**: Semi-supervised sequence learning. 비지도·준지도 시퀀스 학습으로 언어 표현을 학습하는 선행 연구.
- [2] **Peters et al. (2018a)**: ELMo. 문맥을 반영한 깊은 단어 표현(Deep contextualized word representations).
- [3] **Radford et al. (2018)**: OpenAI GPT. 생성적 사전 훈련 트랜스포머(Generative Pre-trained Transformer).
- [4] **Howard and Ruder (2018)**: ULMFiT. 범용 언어 모델 미세 조정(Universal Language Model Fine-tuning for Text Classification).
- [5] **Bowman et al. (2015)**: SNLI. 자연어 추론 데이터셋(Stanford Natural Language Inference).
- [6] **Williams et al. (2018)**: MultiNLI. 다중 장르 자연어 추론 벤치마크.
- [7] **Dolan and Brockett (2005)**: MSR 패러프레이즈 말뭉치. 문장 쌍의 paraphrase 관계 데이터.
- [8] **Tjong Kim Sang and De Meulder (2003)**: CoNLL-2003. 개체명 인식 공유 작업 및 데이터셋.
- [9] **Rajpurkar et al. (2016)**: SQuAD. 스탠퍼드 질의응답 데이터셋(Stanford Question Answering Dataset).

### 2문단

#### 원문
**There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo (Peters et al., 2018a), uses task-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters. The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.**

#### 번역
사전 훈련된 언어 표현을 **하위 작업(downstream task)** 에 쓰는 방법은 두 가지다. **특성 기반(feature-based)** 과 **미세 조정(fine-tuning)** 이다. 특성 기반은 ELMo[^30]처럼, 사전 훈련된 표현을 “추가 입력”으로 넣고 그 위에 작업용 모델을 따로 만든다. 미세 조정은 OpenAI GPT[^31]처럼, 사전 훈련된 모델 전체를 하위 작업 데이터로 조금 더 훈련하는 방식이다. 작업용으로 새로 넣는 파라미터는 최소한만 둔다. 두 방식 모두 사전 훈련할 때는 같은 목적 함수를 쓰고, **단방향(unidirectional)** 언어 모델로 일반적인 표현을 먼저 학습한다.

[^30]: Peters et al., 2018a
[^31]: Radford et al., 2018

#### 각주 (부가 설명)
- **Peters et al. (2018a)**: ELMo. 사전 훈련된 표현을 추가 특성으로 쓰는 특성 기반 접근의 대표 사례.
- **Radford et al. (2018)**: OpenAI GPT. 작업별 파라미터를 최소화하고 전역 미세 조정을 쓰는 생성적 사전 훈련 트랜스포머.

### 3문단

#### 원문
**We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer (Vaswani et al., 2017). Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.**

#### 번역
이 논문의 주장은, 지금 방식이 사전 훈련 표현의 성능을 제한하고 있다는 것이다. 미세 조정 쪽이 특히 그렇다. 이유는 간단하다. 일반적인 언어 모델은 **단방향**이라서, 사전 훈련에 쓸 수 있는 구조가 제한되기 때문이다. 예를 들어 OpenAI GPT는 **왼쪽→오른쪽**만 보는 구조다. 트랜스포머의 **셀프 어텐션(self-attention)** 층에서 각 토큰은 “이전” 토큰만 볼 수 있다[1]. 문장 수준 작업은 그래도 어느 정도 되지만, 질의응답처럼 **앞뒤 문맥을 다 봐야 하는** 토큰 수준 작업에는 이게 큰 약점이 된다.

#### 각주 (부가 설명)
- **[1] Vaswani et al. (2017)**: "Attention Is All You Need". 셀프 어텐션만으로 구성된 트랜스포머(Transformer) 구조를 제안한 논문.

### 4문단

#### 원문
**In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT alleviates the previously mentioned unidirectionality constraint by using a "masked language model" (MLM) pre-training objective, inspired by the Cloze task (Taylor, 1953). The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, we also use a "next sentence prediction" task that jointly pre-trains text-pair representations. The contributions of our paper are as follows:**

#### 번역
이 논문에서는 **BERT**(Bidirectional Encoder Representations from Transformers)를 제안해서 미세 조정 방식을 개선한다. BERT는 **마스크드 언어 모델(Masked Language Model, MLM)** 을 사전 훈련 목표로 쓴다. MLM은 예전부터 쓰이던 빈칸 채우기(Cloze) 작업(Taylor, 1953)와 같은 발상이다. 입력 토큰 중 일부를 무작위로 가리고, **문맥만 보고** 그 자리에 들어갈 단어(어휘 ID)를 맞추게 한다. 왼쪽→오른쪽 언어 모델과 달리, 이렇게 하면 **왼쪽·오른쪽 문맥을 모두 쓴** 표현을 학습할 수 있어서, 진짜 **양방향** 트랜스포머를 사전 훈련할 수 있다. 거기에 **다음 문장 예측(Next Sentence Prediction, NSP)** 작업을 더해서, 문장 두 개가 이어지는지도 함께 학습한다. 이 논문의 기여는 아래 세 가지다.

#### 각주 (부가 설명)
- **Taylor (1953)**: "Cloze procedure". 빈칸 채우기 작업을 제안한 심리측정/언어 연구. MLM의 직간접적 선행 개념.

### 어휘/숙어 설명 (Introduction)

| 원문 | 한글 | 의미 |
|------|------|------|
| **task** | 작업 | 논문에서 말하는 task는 "풀려는 문제", "목표로 하는 작업" 정도의 의미다. 학교에서의 숙제(과제)와는 다르다. |
| **pre-training** | 사전 훈련 | 감정 분석·질의응답 같은 "목표 작업"에 쓰기 전에, **레이블 없이** 대량 텍스트만으로 모델을 먼저 학습하는 단계. 기초를 쌓아 두면 나중에 적은 데이터로도 잘 맞출 수 있다. |
| **pre-trained** | 사전 훈련된 | 위와 같이 사전 훈련을 마친 상태. "이미 대량 데이터로 한 번 학습된" 모델을 가리킨다. |
| **bidirectional pre-training** | 양방향 사전 훈련 | 사전 훈련할 때 **앞뒤 문맥을 모두** 쓰는 방식. 단방향(왼쪽→오른쪽만 보는 GPT 등)과 달리, 각 단어가 "왼쪽만"이 아니라 "왼쪽+오른쪽" 문맥을 함께 보면서 학습한다. 질의응답·개체명 인식처럼 앞뒤가 다 필요한 작업에 유리하다. BERT는 MLM(빈칸 채우기)으로 이걸 한 번에 깊게 학습한다. |
| **downstream tasks** | 하위 작업 | 사전 훈련 후 적용하는 구체적 작업(감정 분석, 질의응답 등). |
| **feature-based / fine-tuning** | 특성 기반 / 미세 조정 | 전자: 사전 훈련 특성을 추출해 쓰는 방식. 후자: 사전 훈련 파라미터를 조정해 쓰는 방식 |
| **task-specific architectures** | 작업 특화 구조 | 특정 작업을 위해 설계된 신경망 구조 |
| **objective function** | 목적 함수 | 학습 시 최소화(또는 최대화)하는 식 |
| **unidirectional** | 단방향 | 한 방향(왼쪽→오른쪽 또는 그 반대)으로만 보는 것 |
| **restrict / limitation** | 제한하다 / 한계 | 잠재력을 좁힘 / 그 경계 |
| **attend to** | ~에 주의를 기울이다 | (어텐션에서) 해당 위치를 참고하다 |
| **self-attention** | 셀프 어텐션 | 문장 내 각 토큰이 서로를 참고하는 메커니즘 |
| **sub-optimal** | 차선의 | 최적은 아니지만 어느 정도 괜찮은 |
| **incorporate** | 통합하다 | (문맥을) 포함·반영하다 |
| **alleviate / constraint** | 완화하다 / 제약 | 정도를 줄이다 / 행동을 막는 조건 |
| **masked language model (MLM)** | 마스크드 언어 모델 | 일부 토큰을 가리고 문맥만으로 예측하는 방식 |
| **Cloze task** | 빈칸 채우기 작업 | 빈칸에 들어갈 단어를 맞추는 작업 |
| **vocabulary id** | 어휘 ID | 어휘에서 단어에 부여한 고유 번호 |
| **jointly pre-trains** | 함께 사전 훈련한다 | (문장 쌍 표현을) 동시에 학습한다 |

---

## Key Contributions (주요 기여)

### 원문
**The contributions of our paper are as follows:**
- **We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT uses masked language models to enable pre-trained deep bidirectional representations. This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs.**

- **We show that pre-trained representations reduce the need for many heavily-engineered task-specific architectures. BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures.**

- **BERT achieves state-of-the-art performance on eleven NLP tasks. Code and pre-trained models are available.**

### 번역
이 논문의 기여는 세 가지다.

- **양방향 사전 훈련이 왜 중요한지 보여준다.** Radford et al. (2018)은 단방향 언어 모델로만 사전 훈련한다. BERT는 마스크드 언어 모델(MLM)을 쓰기 때문에, **깊은** 양방향 표현을 한 번에 학습할 수 있다. Peters et al. (2018a)처럼 왼쪽→오른쪽·오른쪽→왼쪽을 따로 훈련한 뒤 **얕게 이어 붙인(shallow concatenation)** 것과는 다르다.

- **사전 훈련만 잘 되면 작업별로 복잡한 구조가 필요 없다는 걸 보여준다.** BERT는 “미세 조정만 하는” 표현 모델 중에서는 처음으로, 문장 수준·토큰 수준을 묶은 넓은 벤치마크에서 최고 성능을 냈다. 작업마다 따로 설계한 모델들보다 나은 경우가 많다.

- **11개 NLP 작업에서 당시 최고 성능을 달성했다.** 코드와 사전 훈련 모델은 https://github.com/google-research/bert 에서 받을 수 있다.

### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **bidirectional** | 양방향 | 양쪽 방향. 왼쪽과 오른쪽 둘 다 |
| **concatenation** | 연결/결합 | 여러 것을 이어 붙이는 것 |
| **heavily-engineered** | 무겁게 설계된 | 복잡하게 만들어진, 많은 노력과 전문성이 들어간 |
| **suite of tasks** | 작업 모음 | 여러 개의 관련된 작업들 |
| **outperforming** | 능가하다 | 더 좋은 성과를 내다 |
| **state-of-the-art** | 최첨단/최고 수준의 | 현재로서는 가장 좋은 기술이나 성과를 의미하는 용어 |

---

## 2. Related Work (관련 연구)

**원문:** There is a long history of pre-training general language representations, and we briefly review the most widely-used approaches in this section.

**번역:** “일반적인 언어 표현”을 사전 훈련하는 연구는 꽤 오래되었다. 이 절에서는 그중 가장 많이 쓰이는 방법들을 짧게 정리한다.

---

### 2.1 Unsupervised Feature-based Approaches (비지도 특성 기반 접근)

#### 원문
**Learning widely applicable representations of words has been an active area of research for decades, including non-neural (Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006) and neural (Mikolov et al., 2013; Pennington et al., 2014) methods. Pre-trained word embeddings are an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch (Turian et al., 2010). To pre-train word embedding vectors, left-to-right language modeling objectives have been used (Mnih and Hinton, 2009), as well as objectives to discriminate correct from incorrect words in left and right context (Mikolov et al., 2013).**

#### 번역
“여러 작업에 쓸 수 있는 단어 표현”을 만드는 연구는 수십 년째 이어져 왔다. 비신경망 방법[^1]과 신경망 방법[^2]이 모두 여기 포함된다. **사전 훈련된 단어 임베딩**은 지금 NLP에서 빼놓을 수 없는 부품이다. 처음부터 학습한 임베딩보다 성능이 확실히 좋다[^3]. 단어 임베딩을 사전 훈련할 때 쓰는 목표로는, 왼쪽→오른쪽 언어 모델[^4]이나, “좌우 문맥만 보고 맞는 단어 vs 틀린 단어”를 구분하는 목표[^5] 등이 있다.

[^1]: Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006
[^2]: Mikolov et al., 2013; Pennington et al., 2014
[^3]: Turian et al., 2010
[^4]: Mnih and Hinton, 2009
[^5]: Mikolov et al., 2013

#### 원문
**These approaches have been generalized to coarser granularities, such as sentence embeddings (Kiros et al., 2015; Logeswaran and Lee, 2018) or paragraph embeddings (Le and Mikolov, 2014). To train sentence representations, prior work has used objectives to rank candidate next sentences (Jernite et al., 2017; Logeswaran and Lee, 2018), left-to-right generation of next sentence words given a representation of the previous sentence (Kiros et al., 2015), or denoising auto-encoder derived objectives (Hill et al., 2016).**

#### 번역
이런 접근은 **문장** 단위[^6], **단락** 단위[^7]처럼 더 큰 단위(coarser granularities)로도 확장됐다. 문장 표현을 학습하려면 “다음에 올 문장 후보에 순위 매하기”[^8], “이전 문장 표현이 주어졌을 때 다음 문장을 왼쪽→오른쪽으로 생성”[^9], **디노이징 오토인코더**에서 나온 목표[^10] 같은 걸 쓴다.

[^6]: Kiros et al., 2015; Logeswaran and Lee, 2018
[^7]: Le and Mikolov, 2014
[^8]: Jernite et al., 2017; Logeswaran and Lee, 2018
[^9]: Kiros et al., 2015
[^10]: Hill et al., 2016

#### 원문
**ELMo and its predecessor (Peters et al., 2017, 2018a) generalize traditional word embedding research along a different dimension. They extract context-sensitive features from a left-to-right and a right-to-left language model. The contextual representation of each token is the concatenation of the left-to-right and right-to-left representations. When integrating contextual word embeddings with existing task-specific architectures, ELMo advances the state of the art for several major NLP benchmarks (Peters et al., 2018a) including question answering (Rajpurkar et al., 2016), sentiment analysis (Socher et al., 2013), and named entity recognition (Tjong Kim Sang and De Meulder, 2003). Melamud et al. (2016) proposed learning contextual representations through a task to predict a single word from both left and right context using LSTMs. Similar to ELMo, their model is feature-based and not deeply bidirectional. Fedus et al. (2018) shows that the cloze task can be used to improve the robustness of text generation models.**

#### 번역
**ELMo**와 그 전신[^11]은 단어 임베딩을 “다른 축”으로 확장한 케이스다. 왼쪽→오른쪽 언어 모델 하나, 오른쪽→왼쪽 언어 모델 하나를 따로 두고, 각 토큰의 **문맥 표현**은 두 모델의 출력을 **이어 붙인(concatenation)** 벡터로 쓴다. 이 문맥-sensitive 단어 임베딩을 기존 작업용 모델에 넣었더니, 질의응답[^12], 감정 분석[^13], 개체명 인식[^14] 등에서 당시 최고 성능이 나왔다[^15]. Melamud et al.는 LSTM으로 “좌우 문맥 보고 한 단어 맞추기”로 문맥 표현을 학습하는 걸 제안했는데[^16], ELMo처럼 특성 기반이고, **깊은** 양방향은 아니다. Fedus et al.는 클로즈(빈칸 채우기) 작업이 텍스트 생성 모델의 **견고성(robustness)**을 키우는 데 쓸 수 있음을 보였다[^17].

[^11]: Peters et al., 2017, 2018a
[^12]: Rajpurkar et al., 2016
[^13]: Socher et al., 2013
[^14]: Tjong Kim Sang and De Meulder, 2003
[^15]: Peters et al., 2018a
[^16]: Melamud et al., 2016
[^17]: Fedus et al., 2018

#### 어휘/숙어 설명 (2.1)

| 원문 | 한글 | 의미 |
|------|------|------|
| **widely applicable** | 널리 적용 가능한 | 여러 작업이나 상황에 두루 쓰일 수 있는 |
| **integral part** | 필수 구성 요소 | 빠뜨릴 수 없는 부분 |
| **from scratch** | 처음부터 | 사전 학습 없이 제로부터 |
| **discriminate** | 구별하다 | 옳은 것과 틀린 것을 가리다 |
| **coarser granularities** | 더 굵은 입도 | 더 큰 단위(문장·단락 등). 세밀한 단어 단위보다 굵은 단위 |
| **denoising auto-encoder** | 디노이징 오토인코더 | 입력에 잡음을 넣었다가 원래 입력을 복원하도록 학습하는 구조 |
| **context-sensitive** | 문맥에 민감한 | 주변 단어에 따라 의미가 달라지는 |
| **robustness** | 견고성 | 방해 요인이 있어도 잘 견디는 성질 |

---

### 2.2 Unsupervised Fine-tuning Approaches (비지도 미세 조정 접근)

#### 원문
**As with the feature-based approaches, the first works in this direction only pre-trained word embedding parameters from unlabeled text (Collobert and Weston, 2008).**

**More recently, sentence or document encoders which produce contextual token representations have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018). The advantage of these approaches is that few parameters need to be learned from scratch. At least partly due to this advantage, OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentence-level tasks from the GLUE benchmark (Wang et al., 2018a). Left-to-right language modeling and auto-encoder objectives have been used for pre-training such models (Howard and Ruder, 2018; Radford et al., 2018; Dai and Le, 2015).**

#### 번역
특성 기반과 마찬가지로, 이쪽 계열도 처음엔 “비표지 텍스트로 단어 임베딩만” 사전 훈련하는 수준이었다[^32].

[^32]: Collobert and Weston, 2008

그 다음에는 **문맥을 반영한 토큰 표현**을 만드는 문장/문서 **인코더**를 비표지 텍스트로 사전 훈련한 뒤, **지도 학습** 하위 작업에 맞춰 미세 조정하는 방식이 널리 쓰인다[^18]. 장점은 “처음부터 학습할 파라미터”가 적다는 것이다. 그 덕분에 OpenAI GPT[^19]가 GLUE 벤치마크[^20]의 여러 문장 수준 작업에서 당시 최고 성능을 냈다. 이런 모델들 사전 훈련에는 왼쪽→오른쪽 언어 모델 목표나 오토인코더 목표가 쓰인다[^21].

[^18]: Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018
[^19]: Radford et al., 2018
[^20]: Wang et al., 2018a
[^21]: Howard and Ruder, 2018; Radford et al., 2018; Dai and Le, 2015

#### 어휘/숙어 설명 (2.2)

| 원문 | 한글 | 의미 |
|------|------|------|
| **encoder** | 인코더 | 입력을 고정된 크기의 표현(벡터)으로 바꾸는 모듈 |
| **supervised downstream task** | 지도 학습 하위 작업 | 레이블이 있는 데이터로 학습하는 실제 적용 작업 |
| **at least partly due to** | 적어도 부분적으로 ~ 때문에 | 그 이유가 전부는 아니어도 한 원인으로 |

---

### 2.3 Transfer Learning from Supervised Data (지도 데이터에서의 전이 학습)

#### 원문
**There has also been work showing effective transfer from supervised tasks with large datasets, such as natural language inference (Conneau et al., 2017) and machine translation (McCann et al., 2017). Computer vision research has also demonstrated the importance of transfer learning from large pre-trained models, where an effective recipe is to fine-tune models pre-trained with ImageNet (Deng et al., 2009; Yosinski et al., 2014).**

#### 번역
**지도 학습**으로 많이 모은 데이터가 있는 작업(자연어 추론[^33], 기계 번역[^34] 등)에서도 “그걸 사전 훈련처럼 써서 전이한다”는 연구가 있다. 컴퓨터 비전 쪽도 마찬가지다. 큰 데이터로 미리 학습한 모델을 쓰는 **전이 학습**이 효과적이라는 결과가 많고, 그중에서도 ImageNet으로 사전 훈련한 모델을 미세 조정하는 방식이 잘 통한다[^35].

[^33]: Conneau et al., 2017
[^34]: McCann et al., 2017
[^35]: Deng et al., 2009; Yosinski et al., 2014

#### 어휘/숙어 설명 (2.3)

| 원문 | 한글 | 의미 |
|------|------|------|
| **transfer** | 전이 | 한 작업에서 배운 지식을 다른 작업에 옮겨 쓰는 것 |
| **recipe** | (여기서) 방법/비결 | 효과가 검증된 구체적인 절차나 설정 |

---

## 3. BERT (Model Architecture)

### 원문
**BERT's unique feature is a unified architecture across a diverse set of downstream tasks. Minimal differences are introduced between the pre-training architecture and the final downstream task architecture.**

**Model Architecture: The model architecture of BERT is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library. Since the use of Transformers has become ubiquitous and our implementation is nearly identical to the original, we omit tedious background exposition and refer readers to Vaswani et al. (2017) and excellent guides such as "The Annotated Transformer."**

### 번역
BERT의 특징은, 여러 하위 작업에 **똑같은 구조(unified architecture)** 를 쓴다는 것이다. 사전 훈련할 때 쓰는 구조와, 실제 작업(질의응답, 분류 등)에 쓸 때 구조가 거의 같다. 출력층만 갈아끼우면 된다.

**모델 구조:** BERT는 Vaswani et al.[^22]에서 제안한 **트랜스포머**를 그대로 쓰는 **다층 양방향 트랜스포머 인코더**다. `tensor2tensor` 라이브러리에 공개된 구현과 동일하다. 트랜스포머 자체는 이미 널리 알려져 있으니, 여기서는 구조에 대한 긴 설명을 생략하고 Vaswani et al.[^22]이나 "The Annotated Transformer"[^23] 같은 자료를 참고하면 된다.

[^22]: Vaswani et al., 2017
[^23]: "The Annotated Transformer" (Harvard NLP)

### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **unique feature** | 독특한 특징/특징 | 다른 것과는 다른 점, 특별한 점 |
| **unified architecture** | 통합 구조 | 여러 작업에 하나의 같은 구조를 사용하는 것 |
| **multi-layer** | 다층의 | 여러 층으로 이루어진. 신경망의 깊이를 나타냄 |
| **bidirectional Transformer encoder** | 양방향 트랜스포머 인코더 | 양쪽 방향의 정보를 모두 사용하여 텍스트를 분석하는 신경망 |
| **implementation** | 구현 | 이론이나 설계를 실제 코드나 시스템으로 만드는 것 |
| **ubiquitous** | 광범위하게 사용되는/편재적 | 어디에나 있는, 매우 흔한 |
| **omit** | 생략하다 | 빼먹다, 빼버리다 |
| **tedious** | 지루한/번거로운 | 반복되고 지루해 보이는, 힘이 드는 |

---

### 원문
**In this work, we denote the number of layers (Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. We report results on two model sizes:**

- **BERT_BASE: L=12, H=768, A=12, Total Parameters=110M**
- **BERT_LARGE: L=24, H=1024, A=16, Total Parameters=340M**

**BERT_BASE is chosen to have the same model size as OpenAI GPT for comparison purposes, but there is an important difference: the BERT Transformer uses bidirectional self-attention, whereas the GPT Transformer uses restricted self-attention where every token can only attend to previous tokens.**

### 번역
논문에서는 **층 개수**(트랜스포머 블록 수)를 L, **은닉 차원**을 H, **어텐션 헤드 개수**를 A로 표기한다. 실험은 두 크기로 했다.

- **BERT_BASE**: L=12, H=768, A=12, 총 파라미터 110M
- **BERT_LARGE**: L=24, H=1024, A=16, 총 파라미터 340M

BERT_BASE는 OpenAI GPT와 “모델 크기”를 맞춰서 공정 비교하려고 이렇게 잡았다. 대신 **구조**는 다르다. BERT는 **양방향**이라 각 토큰이 앞뒤 문맥을 다 볼 수 있고, GPT는 **한 방향**이라 각 토큰이 “이전” 토큰만 볼 수 있다(제한된 셀프 어텐션).

### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **denote** | 표기하다/나타내다 | 기호나 문자로 나타내다 |
| **hidden size** | 은닉 크기 | 신경망 내부의 각 층이 처리하는 데이터의 차원(크기) |
| **self-attention heads** | 셀프 어텐션 헤드 | 문맥을 분석할 때 여러 각도에서 동시에 분석하는 병렬 처리 단위 |
| **Parameters** | 파라미터 | 신경망이 학습하는 가중치들. 개수가 많을수록 모델이 큼 |
| **restricted self-attention** | 제한된 셀프 어텐션 | 모든 단어를 볼 수 없고 앞의 단어들만 볼 수 있게 제한한 것 |

---

### 3.1 Input/Output Representations (입력/출력 표현)

#### 원문
**To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., 〈 Question, Answer 〉) in one token sequence. Throughout this work, a "sentence" can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together. We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B. As shown in Figure 1, we denote input embedding as E, the final hidden vector of the special [CLS] token as C ∈ R^H, and the final hidden vector for the ith input token as T_i ∈ R^H. For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings.**

#### 번역
BERT가 “한 문장”이든 “문장 두 개”(예: 질문+지문)든 다 받을 수 있게, **입력 표현** 을 통일했다. 하나의 **토큰 시퀀스** 로 넣는다. 여기서 “문장”은 문법상 한 문장이 아니라, **연속된 텍스트의 한 구간** 을 의미할 경우도 있다. “시퀀스”는 BERT에 넣는 **토큰 열 전체** 를 말한다. 30,000 크기 **WordPiece** 어휘[^24]를 쓴다. 시퀀스 맨 앞에는 항상 특수 토큰 **[CLS]** 를 둔다. 분류 작업에서는 이 [CLS] 위치의 **최종 은닉 벡터** 를 “문장(또는 문장 쌍) 전체를 대표하는 벡터”로 쓴다. 문장이 두 개면 **[SEP]** 토큰으로 구분해서 한 시퀀스로 이어 붙이고, 각 토큰이 “문장 A인지 B인지”를 나타내는 **세그먼트 임베딩** 을 더한다. 그림 1처럼 입력 임베딩을 E, [CLS] 최종 벡터를 C ∈ R^H, i번째 토큰 최종 벡터를 T_i ∈ R^H로 둔다. 토큰 하나의 입력 표현 = 토큰 임베딩 + 세그먼트 임베딩 + 위치 임베딩이다.

[^24]: Wu et al., 2016

### Pre-training Data (사전 훈련 데이터)

#### 원문
**The pre-training procedure largely follows the existing literature on language model pre-training. For the pre-training corpus we use the BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark (Chelba et al., 2013) in order to extract long contiguous sequences.**

#### 번역
사전 훈련 절차는 다른 언어 모델 사전 훈련 논문들과 비슷하다. **코퍼스** 는 BooksCorpus(약 8억 단어)[^25]와 영어 위키백과(약 25억 단어)를 썼다. 위키백과는 본문만 추출하고 표·목록·헤더는 제외했다. “긴 연속 구간”을 쓰려면 문장을 다 잘라서 섞어 둔 Billion Word Benchmark[^26] 같은 것보다, **문서 단위** 로 길게 이어진 코퍼스가 맞다.

[^25]: Zhu et al., 2015
[^26]: Chelba et al., 2013

---

## Pre-training Tasks (사전 훈련 작업)

### Task #1: Masked Language Model (MLM)

#### 원문
**Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly "see itself" in a multi-layer context, and thus trivially predict the target word.**

#### 번역
직관적으로 생각하면, “깊은 양방향” 모델이 “한 방향만 보는” 모델이나 “왼쪽→오른쪽 + 오른쪽→왼쪽을 얕게 이어 붙인” 모델보다 더 강할 것 같다. 그런데 **일반적인 조건부 언어 모델** 은 왼쪽→오른쪽 *또는* 오른쪽→왼쪽 **둘 중 하나** 로만 훈련할 수 있다. 양쪽을 동시에 쓰면, 예측 대상 단어가 여러 층을 거치면서 **간접적으로 자기 자신** 을 참고하게 되어, 문제가 너무 쉬워진다(trivially).

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **intuitively** | 직관적으로 | 논리적 설명 없이도 자명하게, 직감적으로 |
| **reasonable** | 합리적인 | 논리적으로 타당한, 말이 되는 |
| **conditional language models** | 조건부 언어 모델 | 앞의 단어들을 조건으로 주고 다음 단어를 예측하는 모델 |
| **bidirectional conditioning** | 양방향 조건화 | 앞뒤 양쪽의 정보를 모두 사용하여 예측하는 것 |
| **indirectly** | 간접적으로 | 직접이 아닌 다른 경로를 통해 |
| **trivially** | 사소하게/쉽게 | 너무 쉬워서 의미가 없는 방식으로 |
| **target word** | 대상 단어 | 예측해야 하는 목표 단어 |

---

#### 원문
**To train a deep bidirectional representation, we simply mask a random sample of the tokens from the input, and then predict those masked tokens. The procedure is called a "Masked LM" (MLM), and is inspired by the Cloze task in literature (Taylor, 1953). In this case, the final hidden vector corresponding to the mask token is fed into an output softmax over the vocabulary, as in a standard language model. All experiments use an MLM where we mask 15% of the WordPiece tokens in each sequence at random.**

#### 번역
그래서 **입력 토큰 중 일부를 무작위로 가리고**, 그 자리에 뭐가 들어갈지 예측하게 한다. 이걸 **마스크드 언어 모델(Masked LM, MLM)** 이라고 부른다. 예전부터 쓰이던 빈칸 채우기(Cloze)[^27]와 같은 발상이다. 마스크된 위치의 **최종 은닉 벡터** 를 받아서, 어휘 크기만큼의 **소프트맥스** 로 원래 토큰을 예측한다. 실험에서는 시퀀스마다 WordPiece 토큰의 **15%** 를 무작위로 마스킹했다.

[^27]: Taylor, 1953

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **mask** | 가리다/마스킹하다 | 어떤 것을 숨기거나 보이지 않게 하다 |
| **random sample** | 무작위 샘플 | 규칙 없이 임의로 선택한 것들 |
| **hidden vector** | 은닉 벡터 | 신경망이 내부적으로 생성하는 숫자 배열 |
| **softmax** | 소프트맥스 | 여러 확률값을 0~1 사이로 정규화하는 수학 함수 |
| **vocabulary** | 어휘/사전 | 모델이 알고 있는 모든 단어들의 목록 |
| **WordPiece tokens** | WordPiece 토큰 | 단어를 부분적으로 나눈 토큰. 예: "running" → "run", "##ning" |

---

#### 원문
**However, this creates a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. To mitigate this, we do not always replace "masked" words with the actual [MASK] token. Instead, the training data generator randomly selects 15% of the token positions in each sequence.**

**If the i-th token is selected, we (1) replace it with [MASK] 80% of the time (2) replace it with a random token 10% of the time, or (3) keep it unchanged 10% of the time.**

**We then use the final hidden vector corresponding to this token to predict the original token with cross-entropy loss.**

#### 번역
다만 미세 조정할 때는 `[MASK]` 토큰이 **안 나온다**. 그래서 사전 훈련과 미세 조정 사이에 **불일치(mismatch)** 가 생길 수 있다. 이걸 줄이려고, “마스킹할 15% 위치”만 정한 뒤, 그 위치를 **항상** [MASK]로 바꾸지는 않는다. i번째 토큰이 뽑히면:

- **80%:** [MASK]로 바꾼다
- **10%:** 다른 **무작위 토큰**으로 바꾼다
- **10%:** **그대로** 둔다

그 위치의 최종 은닉 벡터로 “원래 토큰”을 예측하게 하고, **교차 엔트로피 손실(cross-entropy loss)** 로 학습한다.

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **mismatch** | 불일치/차이 | 두 가지가 서로 맞지 않는 상태 |
| **mitigate** | 완화하다 | 문제의 정도를 줄이거나 심해지는 것을 방지하다 |
| **cross-entropy loss** | 교차 엔트로피 손실 | 예측값과 실제값의 차이를 계산하는 수식 |

---

### Task #2: Next Sentence Prediction (NSP)

#### 원문
**Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train a binarized next sentence prediction task that can be trivially generated from any monolingual corpus.**

**Specifically, when choosing the sentences A and B for each pre-training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext).**

#### 번역
질의응답(QA), 자연어 추론(NLI) 같은 작업은 “문장 두 개의 **관계** ”를 아는 게 중요하다. 그런데 언어 모델링만으로는 이걸 잘 못 배운다. 그래서 **다음 문장 예측(Next Sentence Prediction, NSP)** 이라는 작업을 사전 훈련에 넣었다. 단일 언어 코퍼스만 있어도 쉽게 만들 수 있다. 문장 A와 B를 고를 때, 50%는 B가 A **바로 다음** 문장(레이블 `IsNext`), 50%는 코퍼스에서 **아무 문장** 이나 뽑은 것(레이블 `NotNext`)이다.

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **relationship** | 관계 | 두 사물 간의 연결 또는 상호작용 |
| **binarized** | 이진화된 | 두 가지 선택지만 있는 상태. 예: Yes/No, True/False |
| **monolingual corpus** | 단일 언어 코퍼스 | 한 가지 언어로만 이루어진 텍스트 모음 |
| **trivially generated** | 사소하게 생성된 | 별다른 노력 없이 쉽게 만들어질 수 있는 |
| **actual next sentence** | 실제 다음 문장 | 원문에서 정말로 그 다음에 오는 문장 |
| **random sentence** | 무작위 문장 | 아무 관계없이 무작위로 선택된 문장 |

---

## 3.2 Fine-tuning BERT (미세 조정)

#### 원문
**Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks—whether they involve single text or text pairs—by swapping out the appropriate inputs and outputs. For applications involving text pairs, a common pattern is to independently encode text pairs before applying bidirectional cross attention, such as Parikh et al. (2016); Seo et al. (2017). BERT instead uses the self-attention mechanism to unify these two stages, as encoding a concatenated text pair with self-attention effectively includes bidirectional cross attention between two sentences. For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end. At the input, sentence A and sentence B from pre-training are analogous to (1) sentence pairs in paraphrasing, (2) hypothesis-premise pairs in entailment, (3) question-passage pairs in question answering, and (4) a degenerate text-∅ pair in text classification or sequence tagging. At the output, the token representations are fed into an output layer for token-level tasks, such as sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis. Compared to pre-training, fine-tuning is relatively inexpensive. All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model.**

#### 번역
트랜스포머 **셀프 어텐션** 덕분에 BERT는 "한 문장"이든 "문장 쌍"이든 **입력·출력만** 바꿔서 여러 하위 작업을 같은 구조로 처리할 수 있다. 그래서 미세 조정이 단순하다. 예전에는 문장 쌍을 다룰 때[^28] 문장을 **따로** 인코딩한 뒤 **교차 어텐션** 을 쓰는 경우가 많았다. BERT는 문장 쌍을 **한 시퀀스로 이어 붙인 뒤** 셀프 어텐션으로 한 번에 인코딩하니까, 두 문장 사이 교차 어텐션이 저절로 들어간다. 작업마다 "입력 형식"과 "출력층"만 BERT에 붙이고, **전체 파라미터** 를 끝까지 미세 조정한다. 사전 훈련 때의 "문장 A, 문장 B"는 (1) 패러프레이즈면 문장 쌍, (2) 함의면 가설–전제 쌍, (3) QA면 질문–지문 쌍, (4) 단일 문장 분류/태깅이면 "문장–빈칸" 같은 형태로 대응시킨다. 출력은 토큰 수준(태깅, QA 스팬 등)이면 각 토큰 표현을, **분류**(함의, 감정 등)면 **[CLS]** 표현을 출력층에 넣어 쓴다. 미세 조정은 사전 훈련보다 훨씬 가볍다. 같은 사전 훈련 모델에서 논문 결과 전체를 Cloud TPU 1대로 1시간 안에, GPU로는 몇 시간 안에 재현할 수 있다.

[^28]: Parikh et al., 2016; Seo et al., 2017

---

## Experiments Results (실험 결과)

### GLUE Results

#### 원문
**We fine-tune on GLUE using the representation of the first input token ([CLS]) in the final hidden layer as the aggregate representation of the input. The only new parameters introduced during fine-tuning are classification weights W ∈ ℝ^(K×H), where K is the number of labels. We compute a standard classification loss with log(softmax(CW^T)).**

#### 번역
GLUE 미세 조정에서는 최종 은닉 층의 `[CLS]` 토큰 표현을 입력 전체를 대표하는 벡터로 쓴다. 새로 넣는 파라미터는 분류 가중치 W ∈ ℝ^(K×H)(K는 레이블 수)뿐이고, 손실은 log(softmax(CW^T))로 둔다.

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **aggregate representation** | 집약 표현 | 전체 입력을 하나의 벡터로 요약한 것 |
| **classification weights** | 분류 가중치 | 입력을 특정 범주로 분류할 때 사용하는 수치들 |
| **classification loss** | 분류 손실 | 모델의 예측이 실제값과 얼마나 다른지를 나타내는 수치 |

#### 원문
**We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks. For each task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set. Additionally, for BERT_LARGE we found that fine-tuning was sometimes unstable on small datasets, so we ran several random restarts and selected the best model on the Dev set. With random restarts, we use the same pre-trained checkpoint but perform different fine-tuning data shuffling and classifier layer initialization.**

#### 번역
모든 GLUE 작업에 배치 크기 32, 3 에폭으로 미세 조정했다. 학습률은 개발 세트에서 5e-5, 4e-5, 3e-5, 2e-5 중 가장 좋은 값을 골랐다. BERT_LARGE는 소규모 데이터에서 불안정할 수 있어, 같은 사전 훈련 체크포인트로 데이터 셔플과 분류기 초기화만 바꿔 여러 번 재시작한 뒤 개발 세트 성능이 가장 좋은 모델을 썼다.

---

#### 원문
**BERT_BASE and BERT_LARGE outperform all previous systems by a considerable margin, and achieve an average accuracy increase of 4.5% and 7.0% respectively compared to the prior state-of-the-art. Most notably, BERT_LARGE achieves a score of 80.5 on the official GLUE leaderboard, compared to OpenAI GPT at 72.8.**

#### 번역
BERT_BASE와 BERT_LARGE 모두 기존 시스템들을 상당한 격차로 앞서며, 이전 최고 성능 대비 평균 정확도가 각각 4.5%, 7.0% 올랐다. 특히 BERT_LARGE는 공식 GLUE 리더보드에서 80.5점을 기록했고, 당시 OpenAI GPT는 72.8점이었다.

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **outperform** | 능가하다/뛰어나다 | 다른 것보다 더 좋은 성과를 내다 |
| **considerable margin** | 상당한 차이 | 큰 폭의 차이 |
| **leaderboard** | 리더보드 | 성과 순위를 표시하는 순위표 |

---

## 4.2 SQuAD v1.1

#### 원문
**The Stanford Question Answering Dataset (SQuAD v1.1) is a collection of 100k crowd-sourced question/answer pairs (Rajpurkar et al., 2016). Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage. As shown in Figure 1, in the question answering task, we represent the input question and passage as a single packed sequence, with the question using the A embedding and the passage using the B embedding. We only introduce a start vector S ∈ R^H and an end vector E ∈ R^H during fine-tuning. The probability of word i being the start of the answer span is computed as a dot product between T_i and S followed by a softmax over all of the words in the paragraph. The analogous formula is used for the end of the answer span. The score of a candidate span from position i to position j is defined as S·T_i + E·T_j, and the maximum scoring span where j ≥ i is used as a prediction. The training objective is the sum of the log-likelihoods of the correct start and end positions. We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32.**

#### 번역
SQuAD v1.1[^36]은 10만 개의 크라우드소싱 질문–답변 쌍이다. 위키백과 지문과 질문이 주어지면 지문 안의 답 구간(시작·끝 위치)을 예측한다. 질문은 A, 지문은 B 임베딩으로 넣어 하나의 시퀀스로 표현한다(그림 1). 미세 조정 시 시작 벡터 S, 끝 벡터 E ∈ R^H만 추가하고, 단어 i가 답의 시작일 확률은 T_i·S에 지문 전체에 대한 소프트맥스를 씌워 구한다. 끝 위치도 동일하다. 구간 [i, j]의 점수는 S·T_i + E·T_j로 두고, j ≥ i인 구간 중 점수 최대인 것을 답으로 쓴다. 손실은 정답 시작·끝 위치의 로그 우도 합이다. 학습률 5e-5, 배치 32, 3 에폭으로 미세 조정했다.

[^36]: Rajpurkar et al., 2016

#### 원문
**Table 2 shows top leaderboard entries as well as results from top published systems. Our best performing system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. In fact, our single BERT model outperforms the top ensemble system in terms of F1 score. Without TriviaQA fine-tuning data, we only lose 0.1-0.4 F1, still outperforming all existing systems by a wide margin.**

#### 번역
표 2에 리더보드 상위 및 주요 공개 결과가 있다. 우리 모델은 앙상블 기준 F1 +1.5, 단일 모델 기준 +1.3으로 당시 1위를 넘어섰다. 단일 BERT만으로도 F1 기준 최고 앙상블을 앞섰고, TriviaQA 미세 조정 없이도 F1이 0.1~0.4만 줄어들며 기존 시스템들을 크게 웃돌았다.

---

## 4.3 SQuAD v2.0

#### 원문
**The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic. We use a simple approach to extend the SQuAD v1.1 BERT model for this task. We treat questions that do not have an answer as having an answer span with start and end at the [CLS] token. The probability space for the start and end answer span positions is extended to include the position of the [CLS] token. For prediction, we compare the score of the no-answer span: s_null = S·C + E·C to the score of the best non-null span. We predict a non-null answer when the best span score exceeds s_null + τ, where the threshold τ is selected on the dev set to maximize F1. We did not use TriviaQA data for this model. We fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48.**

#### 번역
SQuAD 2.0은 지문에 답이 없을 수 있는 경우를 허용해 1.1을 확장한, 더 현실적인 설정이다. v1.1 BERT를 그대로 쓰되, 답이 없는 질문은 시작·끝이 모두 [CLS]인 구간으로 간주하고, 시작·끝 후보에 [CLS] 위치를 포함시킨다. 예측 시 무답 점수 s_null = S·C + E·C와 최고 점수 구간을 비교해, 최고 점수가 s_null + τ보다 크면 답이 있다고 예측한다. τ는 개발 세트 F1을 최대화하도록 정했다. TriviaQA는 쓰지 않았고, 학습률 5e-5, 배치 48, 2 에폭으로 미세 조정했다.

#### 원문
**The results compared to prior leaderboard entries and top published work are shown in Table 3. We observe a +5.1 F1 improvement over the previous best system.**

#### 번역
표 3에 이전 리더보드·주요 공개 결과와의 비교가 있다. 이전 최고 시스템보다 F1이 5.1pt 향상되었다.

---

## 4.4 SWAG

#### 원문
**The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded common-sense inference (Zellers et al., 2018). Given a sentence, the task is to choose the most plausible continuation among four choices. When fine-tuning on the SWAG dataset, we construct four input sequences, each containing the concatenation of the given sentence (sentence A) and a possible continuation (sentence B). The only task-specific parameters introduced is a vector whose dot product with the [CLS] token representation C denotes a score for each choice which is normalized with a softmax layer. We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16. Results are presented in Table 4. BERT_LARGE outperforms the authors' baseline ESIM+ELMo system by +27.1% and OpenAI GPT by 8.3%.**

#### 번역
SWAG[^37]는 11만 3천 개 문장 쌍 완성 예시로, 근거 있는 상식 추론을 평가한다. 문장이 주어지면 네 후보 중 가장 자연스러운 이어짐을 고르는 작업이다. 미세 조정 시 (문장 A, 후보 B)를 이어 붙인 시퀀스를 네 개 만들고, [CLS] 표현 C와의 내적으로 각 선택지 점수를 낸 뒤 소프트맥스한다. 작업별 파라미터는 이 점수 벡터 하나뿐이다. 학습률 2e-5, 배치 16, 3 에폭으로 미세 조정했고, 결과는 표 4에 있다. BERT_LARGE는 논문 기준선 ESIM+ELMo보다 27.1%p, OpenAI GPT보다 8.3%p 높았다.

[^37]: Zellers et al., 2018

---

## Ablation Studies (절삭 연구)

### 원문
**In this section, we perform ablation experiments to understand the relative importance of different aspects of BERT. We use BERT_BASE to avoid expensive experiments.**

**First, we evaluate two pre-training objectives by varying the training approach while keeping the training data, fine-tuning methodology, and all other hyperparameters identical to BERT_BASE:**

- **No NSP**: Bidirectional model trained with MLM but without the NSP task.
- **LTR & No NSP**: Left-to-right language model (like GPT) without the NSP task.

#### 번역
이 절에서는 BERT의 각 요소가 얼마나 기여하는지 보기 위해 절삭 실험(ablation)을 한다. 비용을 줄이기 위해 BERT_BASE만 쓴다.

훈련 데이터·미세 조정 방식·하이퍼파라미터는 BERT_BASE와 같게 두고, 사전 훈련 방식만 바꾼 두 가지를 비교한다.

- **No NSP**: MLM으로만 훈련한 양방향 모델(다음 문장 예측 NSP 없음).
- **LTR & No NSP**: MLM 대신 표준 왼쪽→오른쪽(LTR) 언어 모델로 훈련한 모델. NSP도 없다.

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **ablation experiments** | 절삭 실험 | 각 부분을 하나씩 제거하면서 그 부분의 중요도를 측정하는 실험 |
| **relative importance** | 상대적 중요도 | 여러 요소들 중에서 각각이 얼마나 중요한지의 정도 |
| **varying** | 변화시키다 | 다르게 하다, 바꾸다 |
| **hyperparameters** | 하이퍼파라미터 | 모델을 훈련할 때 사람이 미리 설정하는 값들 |

---

#### 원문
**The results show that removing the NSP task hurts performance significantly on QNLI, MNLI, and SQuAD 1.1. Next, we evaluate the importance of bidirectional conditioning by comparing "No NSP" (which uses bidirectional MLM) to "LTR & No NSP" (which uses left-to-right LM).**

**The LTR model performs worse on all tasks compared to the MLM model, and the difference is particularly large on MRPC and SQuAD. These results demonstrate the importance of the MLM pre-training task and bidirectional representations.**

#### 번역
NSP를 빼면 QNLI, MNLI, SQuAD 1.1에서 성능이 뚜렷이 떨어진다. 다음으로 "No NSP"(양방향 MLM)와 "LTR & No NSP"(한 방향 LM)를 비교해 양방향 조건화의 효과를 본다.

LTR 모델은 모든 작업에서 MLM보다 낮고, MRPC와 SQuAD에서 특히 차이가 크다. 이는 MLM 사전 훈련과 양방향 표현이 모두 중요함을 보여준다.

#### 어휘/숙어 설명

| 원문 | 한글 | 의미 |
|------|------|------|
| **hurts performance** | 성능이 떨어지다 | 좋지 않은 영향을 미치다, 나빠지게 하다 |
| **significantly** | 유의미하게 | 통계적으로 의미 있을 정도로, 상당히 |
| **particularly** | 특히/특별히 | 다른 경우보다 더 두드러지게 |

#### 원문
**For SQuAD it is intuitively clear that a LTR model will perform poorly at token predictions, since the token-level hidden states have no right-side context. In order to make a good faith attempt at strengthening the LTR system, we added a randomly initialized BiLSTM on top. This does significantly improve results on SQuAD, but the results are still far worse than those of the pre-trained bidirectional models. The BiLSTM hurts performance on the GLUE tasks. We recognize that it would also be possible to train separate LTR and RTL models and represent each token as the concatenation of the two models, as ELMo does. However: (a) this is twice as expensive as a single bidirectional model; (b) this is non-intuitive for tasks like QA, since the RTL model would not be able to condition the answer on the question; (c) it is strictly less powerful than a deep bidirectional model, since it can use both left and right context at every layer.**

#### 번역
SQuAD에서 LTR이 토큰 예측에 약한 것은, 토큰별 은닉 상태에 오른쪽 문맥이 없기 때문으로 직관적이다. 공정 비교를 위해 LTR 위에 무작위 초기화 BiLSTM을 올려 보았고, SQuAD는 많이 나아졌지만 사전 훈련 양방향 모델에는 미치지 못했다. BiLSTM을 쓰면 GLUE 성능은 오히려 떨어진다. ELMo처럼 LTR과 RTL을 따로 훈련해 이어 붙이는 방법도 있지만, (a) 단일 양방향 모델의 두 배 비용이 들고, (b) QA처럼 질문을 조건으로 답을 내는 구조와 맞지 않으며, (c) 매 층에서 좌우 문맥을 모두 쓰는 깊은 양방향 모델보다 표현력이 낮다.

---

### 5.2 Effect of Model Size (모델 크기의 영향)

#### 원문
**In this section, we explore the effect of model size on fine-tuning task accuracy. We trained a number of BERT models with a differing number of layers, hidden units, and attention heads, while otherwise using the same hyperparameters and training procedure as described previously. Results on selected GLUE tasks are shown in Table 6. In this table, we report the average Dev Set accuracy from 5 random restarts of fine-tuning. We can see that larger models lead to a strict accuracy improvement across all four datasets, even for MRPC which only has 3,600 labeled training examples, and is substantially different from the pre-training tasks. It is also perhaps surprising that we are able to achieve such significant improvements on top of models which are already quite large relative to the existing literature.**

#### 번역
이 절에서는 모델 크기가 미세 조정 성능에 미치는 영향을 본다. 층 수·은닉 크기·어텐션 헤드 수만 바꾼 여러 BERT를, 나머지 설정은 동일하게 두고 훈련했다. 표 6에 선택 GLUE 작업 결과가 있으며, 미세 조정을 5번 무작위 재시작한 뒤 개발 세트 정확도 평균을 썼다. 네 데이터셋 모두에서 더 큰 모델이 꾸준히 더 좋고, 훈련 예시가 3,600개뿐인 MRPC처럼 사전 훈련 작업과도 다른 설정에서도 같은 경향이다. 이미 큰 모델을 더 키웠을 때도 이만큼 오른다는 점은 주목할 만하다.

#### 원문
**It has long been known that increasing the model size will lead to continual improvements on large-scale tasks such as machine translation and language modeling. However, we believe that this is the first work to demonstrate convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained.**

#### 번역
모델을 키우면 기계 번역·언어 모델링 같은 대규모 작업에서 계속 나아진다는 것은 이미 알려져 있었다. 본 논문은 그와 별개로, 충분히 사전 훈련된 모델을 극단적으로 키우면 데이터가 적은 소규모 작업에서도 큰 향상이 난다는 점을 설득력 있게 보여준 초기 사례로 볼 수 있다.

---

### 5.3 Feature-based Approach with BERT (BERT를 이용한 특성 기반 접근)

#### 원문
**All of the BERT results presented so far have used the fine-tuning approach, where a simple classification layer is added to the pre-trained model, and all parameters are jointly fine-tuned on a downstream task. However, the feature-based approach, where fixed features are extracted from the pre-trained model, has certain advantages. First, not all tasks can be easily represented by a Transformer encoder architecture, and therefore require a task-specific model architecture to be added. Second, there are major computational benefits to pre-compute an expensive representation of the training data once and then run many experiments with cheaper models on top of this representation. In this section, we compare the two approaches by applying BERT to the CoNLL-2003 Named Entity Recognition (NER) task. We use the representation of the first sub-token as the input to the token-level classifier over the NER label set. To ablate the fine-tuning approach, we apply the feature-based approach by extracting the activations from one or more layers without fine-tuning any parameters of BERT. These contextual embeddings are used as input to a randomly initialized two-layer 768-dimensional BiLSTM before the classification layer. Results are presented in Table 7. BERT_LARGE performs competitively with state-of-the-art methods. The best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer, which is only 0.3 F1 behind fine-tuning the entire model. This demonstrates that BERT is effective for both fine-tuning and feature-based approaches.**

#### 번역
지금까지의 BERT 결과는 모두 미세 조정 방식이었다. 사전 훈련 모델 위에 분류층을 올리고 전체를 하위 작업에 맞춘다. 한편 사전 훈련 모델을 고정하고 특성만 뽑아 쓰는 특성 기반 접근도 장점이 있다. 작업에 따라 트랜스포머 인코더만으로는 표현이 어려워 작업별 구조가 필요할 수 있고, 비싼 표현을 한 번만 계산한 뒤 그 위에 가벼운 모델로 여러 실험을 할 수 있어 계산이 유리하다. 이 절에서는 BERT를 CoNLL-2003 NER에 적용해 두 방식을 비교한다. 각 토큰의 첫 부분 토큰 표현을 NER 토큰 분류기 입력으로 쓰고, BERT는 전혀 미세 조정하지 않고 한 층 이상에서 활성화만 추출한다. 이 문맥 임베딩을 무작위 초기화 2층 768차원 BiLSTM과 분류층에 넣었다. 표 7에 따르면 BERT_LARGE는 최고 수준 방법과 맞먹고, 상위 4개 은닉층 표현을 이어 붙인 경우가 전체 미세 조정보다 F1이 0.3만 낮다. BERT가 미세 조정과 특성 기반 접근 모두에 잘 맞음을 보여준다.

---

## 6. Conclusion (결론)

#### 원문
**Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from deep unidirectional architectures. Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.**

#### 번역
언어 모델을 이용한 **전이 학습**이 최근 크게 나서면서, “풍부한 비지도 사전 훈련”이 언어 이해 시스템에 꼭 필요하다는 게 드러났다. 덕분에 데이터가 적은 **저자원** 작업도 깊은 단방향 모델의 혜택을 받을 수 있게 됐다. 이 논문의 기여는 그걸 **깊은 양방향** 구조까지 확장한 것이다. 하나의 사전 훈련 모델로 여러 NLP 작업을 잘 풀 수 있게 했다.

---

## Appendix (부록)

부록은 세 부분으로 구성된다: BERT 추가 구현 세부(Appendix A), 실험 추가 세부(Appendix B), 추가 절삭 연구(Appendix C). Appendix C에서는 훈련 스텝 수의 영향, 서로 다른 마스킹 절차에 대한 절삭을 다룬다.

---

### A. Additional Details for BERT (BERT 추가 세부)

#### A.1 Illustration of the Pre-training Tasks (사전 훈련 작업 예시)

**원문:** Assuming the unlabeled sentence is *my dog is hairy*, and during the random masking procedure we chose the 4-th token (which corresponding to *hairy*), our masking procedure can be further illustrated by: 80% of the time replace the word with the [MASK] token; 10% of the time replace with a random word; 10% of the time keep the word unchanged. The purpose of the last is to bias the representation towards the actual observed word. The advantage of this procedure is that the Transformer encoder does not know which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. Additionally, because random replacement only occurs for 1.5% of all tokens (10% of 15%), this does not seem to harm the model's language understanding capability.

**번역:** 비표지 문장 *my dog is hairy*에서 4번째 토큰(*hairy*)이 마스킹 후보로 뽑혔다면, 80%는 [MASK]로, 10%는 무작위 단어로, 10%는 그대로 둔다. 그대로 두는 경우는 표현이 실제 단어에 치우치게 하려는 것이다. 이렇게 하면 인코더는 어떤 위치를 예측할지·어떤 위치가 무작위인지 알 수 없어, 모든 토큰에 대해 문맥 표현을 유지하게 된다. 무작위 대체는 전체의 1.5%(15%×10%)뿐이라 언어 이해 능력을 해치지 않는 것으로 보인다.

**원문 (Next Sentence Prediction):** The next sentence prediction task can be illustrated by: Input = [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP], Label = IsNext; Input = [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP], Label = NotNext.

**번역:** 다음 문장 예측 작업 예시: 입력이 [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]이면 레이블 IsNext, 입력이 [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]이면 레이블 NotNext.

---

#### A.2 Pre-training Procedure (사전 훈련 절차)

**원문:** To generate each training input sequence, we sample two spans of text from the corpus, which we refer to as "sentences" even though they are typically much longer than single sentences. The first sentence receives the A embedding and the second receives the B embedding. 50% of the time B is the actual next sentence that follows A and 50% of the time it is a random sentence. They are sampled such that the combined length is ≤ 512 tokens. The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15%. We train with batch size of 256 sequences (256 × 512 = 128,000 tokens/batch) for 1,000,000 steps (approximately 40 epochs over the 3.3 billion word corpus). We use Adam with learning rate 1e-4, β1=0.9, β2=0.999, L2 weight decay 0.01, learning rate warmup over the first 10,000 steps, and linear decay. Dropout 0.1 on all layers. We use GELU activation (Hendrycks and Gimpel, 2016) rather than ReLU, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood. BERT_BASE was trained on 4 Cloud TPUs (16 TPU chips); BERT_LARGE on 16 Cloud TPUs (64 chips). Each pre-training took 4 days. To speed up pre-training, we use sequence length 128 for 90% of the steps and 512 for the remaining 10% to learn positional embeddings.

**번역:** 각 훈련 샘플은 코퍼스에서 뽑은 텍스트 구간 두 개("문장", 실제로는 한 문장보다 길 수 있음)를 A·B 임베딩으로 이어 붙인다. B는 50% 확률로 A의 실제 다음 문장, 50%로 무작위 문장이다. 두 구간 합이 512 토큰 이하가 되게 샘플링한다. WordPiece 토큰화 후 15% 균일 마스킹을 적용하고, 배치 256 시퀀스(128K 토큰/배치)로 1M 스텝(약 33억 단어, 40 에폭 상당) 훈련한다. Adam(학습률 1e-4, β1=0.9, β2=0.999, L2 0.01), 처음 10K 스텝 워밍업 후 선형 감쇠, 모든 층 드롭아웃 0.1, ReLU 대신 GELU[^38]. 손실은 마스크드 LM 우도와 다음 문장 예측 우도의 평균 합이다. BERT_BASE는 TPU 4대(16칩), BERT_LARGE는 16대(64칩)에서 각 4일 걸렸고, 속도 절약을 위해 90% 스텝은 길이 128, 10%는 512로 위치 임베딩을 학습했다.

[^38]: Hendrycks and Gimpel, 2016

---

#### A.3 Fine-tuning Procedure (미세 조정 절차)

**원문:** For fine-tuning, most model hyperparameters are the same as in pre-training, except batch size, learning rate, and number of training epochs. Dropout was always 0.1. The optimal values are task-specific; the following ranges worked well: Batch size 16 or 32; Learning rate (Adam) 5e-5, 3e-5, or 2e-5; Number of epochs 2, 3, or 4. Large data sets (e.g. 100k+ labeled examples) were far less sensitive to hyperparameter choice than small data sets. Fine-tuning is typically very fast, so it is reasonable to run an exhaustive search over these parameters and choose the best on the development set.

**번역:** 미세 조정 시 배치 크기·학습률·에폭 수만 바꾸고 나머지 하이퍼파라미터는 사전 훈련과 같게 둔다. 드롭아웃 0.1. 실험적으로 잘 동작한 범위는 배치 16 또는 32, 학습률(Adam) 5e-5·3e-5·2e-5, 에폭 2·3·4다. 10만 개 이상 같은 대규모 데이터는 소규모보다 하이퍼파라미터에 덜 민감하다. 미세 조정이 빠르므로 위 범위를 넓게 돌려 보고 개발 세트에서 최선을 고르는 것이 현실적이다.

---

#### A.4 Comparison of BERT, ELMo, and OpenAI GPT (BERT·ELMo·OpenAI GPT 비교)

**원문:** Figure 3 shows the differences. BERT and OpenAI GPT are fine-tuning approaches; ELMo is feature-based. The most comparable method to BERT is OpenAI GPT (left-to-right Transformer LM). Many design decisions in BERT were made to keep it close to GPT for minimal comparison. The core argument is that bidirectionality and the two pre-training tasks account for most of the improvement; other differences include: GPT is trained on BooksCorpus (800M words); BERT on BooksCorpus + Wikipedia (2,500M words). GPT introduces [SEP] and [CLS] only at fine-tuning; BERT learns them and sentence A/B embeddings during pre-training. GPT 1M steps, batch 32k words; BERT 1M steps, batch 128k words. GPT uses 5e-5 for all fine-tuning; BERT selects task-specific learning rate on the dev set. Section 5.1 ablations show that most improvements come from the two pre-training tasks and bidirectionality.

**번역:** 그림 3에서 차이가 나온다. BERT와 OpenAI GPT는 미세 조정 방식, ELMo는 특성 기반이다. BERT와 가장 비슷한 기존 방법은 OpenAI GPT(왼쪽→오른쪽 트랜스포머 LM)이다. BERT의 많은 설계 선택은 GPT와 최소한으로만 비교할 수 있도록 의도적으로 맞춘 것이다. 핵심 주장은 양방향성과 두 사전 훈련 작업이 대부분의 개선을 설명한다는 것이고, 그 외 차이로는: GPT는 BooksCorpus(8억 단어)만, BERT는 BooksCorpus+Wikipedia(25억 단어); GPT는 [SEP]·[CLS]를 미세 조정 시에만 도입하고, BERT는 사전 훈련 시 [SEP], [CLS], 문장 A/B 임베딩을 학습; GPT는 1M 스텝·배치 32k 단어, BERT는 1M 스텝·배치 128k 단어; GPT는 모든 미세 조정에 5e-5, BERT는 개발 세트에서 작업별 학습률 선택. 5.1절 절삭은 개선의 대부분이 두 사전 훈련 작업과 양방향성에서 옴을 보여준다.

---

#### A.5 Illustrations of Fine-tuning on Different Tasks (작업별 미세 조정 그림)

**원문:** Figure 4 shows fine-tuning BERT on different tasks. Task-specific models add one output layer to BERT, so few parameters are learned from scratch. (a)(b) are sequence-level, (c)(d) token-level. E = input embedding, T_i = contextual representation of token i, [CLS] = classification output symbol, [SEP] = separator for non-consecutive token sequences.

**번역:** 그림 4는 작업별 BERT 미세 조정을 보여준다. 작업별 모델은 BERT에 출력층 하나를 더한 것이어서 처음부터 학습하는 파라미터가 적다. (a)(b)는 시퀀스 수준, (c)(d)는 토큰 수준이다. E=입력 임베딩, T_i=토큰 i의 문맥 표현, [CLS]=분류 출력용 특수 심볼, [SEP]=비연속 토큰 시퀀스 구분자.

---

### B. Detailed Experimental Setup (실험 설정 상세)

#### B.1 GLUE Benchmark (GLUE 벤치마크)

**원문/번역 요약:** GLUE 결과는 gluebenchmark.com/leaderboard 등에서 가져왔다. 데이터셋 설명(Wang et al., 2018a 요약): **MNLI** – 대규모 함의 분류, 문장 쌍에 대해 두 번째 문장이 첫 번째에 대해 함의/모순/중립인지 예측. **QQP** – Quora 질문 쌍이 의미적으로 동등한지 이진 분류. **QNLI** – SQuAD를 이진 분류로 변환, (질문, 문장) 쌍이 정답 포함 여부. **SST-2** – 영화 리뷰 문장의 감정 이진 분류. **CoLA** – 영어 문장의 언어적 "수용 가능성" 이진 분류. **STS-B** – 문장 쌍의 의미 유사도 1~5점. **MRPC** – 뉴스에서 추출한 문장 쌍의 의미 동등 여부. **RTE** – MNLI와 유사한 이진 함의, 훈련 데이터 적음. **WNLI** – Winograd NLI 소규모 데이터셋. GLUE 웹페이지에 따르면 WNLI 구성에 문제가 있어 제출된 시스템이 모두 다수 클래스 예측 기준선 65.1보다 낮았다. 공정 비교를 위해 이 세트는 제외했고, GLUE 제출 시에는 항상 다수 클래스를 예측했다.

---

### C. Additional Ablation Studies (추가 절삭 연구)

#### C.1 Effect of Number of Training Steps (훈련 스텝 수의 영향)

**원문:** Figure 5 shows MNLI Dev accuracy after fine-tuning from a checkpoint pre-trained for k steps. (1) Does BERT really need such large pre-training (128k words/batch × 1M steps) for high fine-tuning accuracy? Yes—BERT_BASE gains almost 1.0% on MNLI at 1M vs 500k steps. (2) Does MLM converge slower than LTR since only 15% of words are predicted per batch? The MLM model does converge slightly slower than the LTR model. However, in absolute accuracy the MLM model begins to outperform the LTR model almost immediately.

**번역:** 그림 5는 k 스텝 사전 훈련된 체크포인트에서 미세 조정한 뒤의 MNLI 개발 정확도를 보여준다. (1) BERT가 높은 미세 조정 정확도를 위해 그만큼 큰 사전 훈련(배치당 128k 단어×1M 스텝)이 정말 필요한가? 그렇다. BERT_BASE는 500k 대비 1M 스텝에서 MNLI 약 1.0% 추가 정확도를 얻는다. (2) 배치당 15% 단어만 예측하므로 MLM이 LTR보다 수렴이 느린가? MLM은 LTR보다 약간 더 느리게 수렴한다. 다만 절대 정확도에서는 MLM이 거의 즉시 LTR을 앞선다.

---

#### C.2 Ablation for Different Masking Procedures (마스킹 절차별 절삭)

**원문:** Section 3.1 describes BERT's mixed masking strategy. We evaluate different masking strategies. The goal is to reduce pre-training/fine-tuning mismatch since [MASK] never appears at fine-tuning. We report Dev results for MNLI and NER (fine-tuning and feature-based for NER). Table 8: MASK = replace with [MASK]; SAME = keep token; RND = replace with random token. Left columns are strategy probabilities (BERT uses 80% MASK, 10% SAME, 10% RND). Fine-tuning is surprisingly robust to different masking strategies. Using only MASK was problematic for the feature-based approach on NER. Using only RND performs much worse than our strategy.

**번역:** 3.1절의 BERT 혼합 마스킹(80% MASK, 10% 유지, 10% 무작위)이 미세 조정 시 [MASK]가 없는 상황과의 불일치를 얼마나 줄이는지, 다른 전략과 비교해 절삭한다. 표 8에 MNLI·NER 개발 세트 결과가 있고(NER은 미세 조정·특성 기반 모두), MASK만 쓰면 NER 특성 기반에서 문제가 있었고, 무작위만 쓰면 우리 전략보다 성능이 크게 낮았다. 미세 조정은 마스킹 전략에 상당히 견고했다.

---

## 용어 정리

| 숙어 | 의미 | 예시 |
|------|------|------|
| **state-of-the-art** | 최첨단의, 가장 좋은 수준의 | "This model achieves state-of-the-art performance" = 이 모델은 최고 수준의 성능을 낸다 |
| **jointly conditioning on** | ~에 동시에 의존하다 | "jointly conditioning on left and right context" = 왼쪽과 오른쪽 문맥에 동시에 의존하다 |
| **fine-tuning** | (사전 훈련된 모델의 가중치를) 미세 조정하다 | 대부분의 가중치는 고정하고 작은 폭으로만 조정 |
| **downstream tasks** | (사전 훈련 후) 실제로 수행할 구체적인 작업들 | 감정 분석, 질문 답변 등 |
| **trade-off** | (한쪽을 얻으면 다른 쪽을 잃는) 상충 관계 | speed와 accuracy 사이의 tradeoff |
| **leverage** | ~을 활용하다 | "leverage pre-trained models" = 사전 훈련된 모델을 활용하다 |
| **prohibitively expensive** | (비용이) 너무 비싸서 할 수 없는 | The computation is prohibitively expensive |

---

**문서 구성:** 본 논문(1~16페이지) 및 부록(Appendix A~C)의 원문·한국어 번역, 어휘/숙어 설명표, 고등학생이 모를 만한 숙어 정리를 포함한다.

