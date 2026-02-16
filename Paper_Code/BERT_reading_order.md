# BERT 논문 연구용 읽기 우선순위

BERT 논문(2019-BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding_번역.md)에서 주로 설명하는 내용별로, 참조하는 이전 이론과 **읽기 우선순위**를 정리한 문서이다.

---

## 읽기 우선순위 요약

| 순서 | 목적 | 참조 논문 | 비고 |
|------|------|-----------|------|
| 1 | BERT 구조(인코더, 셀프 어텐션) 이해 | Vaswani et al. (2017) – Transformer | 필수 |
| 2 | BERT가 ELMo/GPT와 어떻게 다른지 | Peters et al. (2018a) – ELMo, Radford et al. (2018) – GPT | 필수 |
| 3 | 사전 훈련/미세 조정 흐름 | Dai and Le (2015), Howard and Ruder (2018), GPT | 권장 |
| 4 | MLM 아이디어의 출처 | Taylor (1953) – Cloze | 선택 |
| 5 | 토크나이저/코퍼스 | Wu et al. (2016) – WordPiece, Zhu et al. (2015) – BooksCorpus | 선택 |

---

## 1. BERT가 “무엇과 다른지” (서론·초록)

**논문이 말하는 것:** BERT는 **양방향** 사전 훈련이고, **미세 조정만**으로 여러 작업에 쓴다.

**참고할 선행 연구:**
- **Peters et al. (2018a)** – ELMo. 문맥 반영 단어 표현, 좌/우 LM을 따로 써서 이어 붙임(특성 기반).
- **Radford et al. (2018)** – OpenAI GPT. 트랜스포머 기반, **단방향** LM 사전 훈련 + 미세 조정.

---

## 2. “사전 훈련·미세 조정”이란 (서론)

**논문이 말하는 것:** 언어 모델 사전 훈련이 NLP에 도움이 되고, 쓰는 방식은 **특성 기반** vs **미세 조정** 두 가지다.

**참고할 선행 연구:**
- **Dai and Le (2015)** – 준지도 시퀀스 학습.
- **Howard and Ruder (2018)** – ULMFiT. 언어 모델 미세 조정.
- **Bowman et al. (2015)** – SNLI. **Williams et al. (2018)** – MultiNLI. (문장 수준 작업 예시.)
- **Tjong Kim Sang and De Meulder (2003)** – CoNLL NER. **Rajpurkar et al. (2016)** – SQuAD. (토큰/스팬 수준 작업 예시.)

---

## 3. “단방향의 한계” (서론 3문단)

**논문이 말하는 것:** GPT처럼 **왼쪽→오른쪽만** 보면 문장/토큰 작업에 불리하다.

**참고할 선행 연구:**
- **Vaswani et al. (2017)** – Transformer. 셀프 어텐션 구조. GPT가 “이전 토큰만 보는 제한된 셀프 어텐션”을 쓰는 배경.

---

## 4. 관련 연구 전체 (2장 Related Work)

**논문이 말하는 것:** 단어·문장·문단 표현, 특성 기반·미세 조정·지도 전이까지 흐름을 짧게 정리.

**참고할 선행 연구 (역사 순):**
- 단어 임베딩: Brown et al., Ando and Zhang, Blitzer et al.; Mikolov et al. (Word2Vec), Pennington et al. (GloVe); Turian et al.; Mnih and Hinton.
- 문장/문단: Kiros et al., Logeswaran and Lee, Le and Mikolov (Doc2Vec); Jernite et al.; Hill et al.
- 문맥 반영: Peters et al. (ELMo); Melamud et al.; Fedus et al.
- 미세 조정 계열: Collobert and Weston (2008); Dai and Le, Howard and Ruder, Radford et al. (GPT); Wang et al. (GLUE).
- 지도 전이: Conneau et al., McCann et al.; Deng et al., Yosinski et al. (ImageNet·전이 학습).

---

## 5. 모델 구조 (3장)

**논문이 말하는 것:** BERT = **다층 양방향 Transformer 인코더**다.

**참고할 선행 연구:**
- **Vaswani et al. (2017)** – Transformer 원 논문. “The Annotated Transformer” 등 가이드도 논문에서 권함.

---

## 6. 입력 표현 (WordPiece, [CLS], [SEP])

**논문이 말하는 것:** 한/두 문장을 하나의 토큰 시퀀스로 넣고, WordPiece·[CLS]·[SEP]·세그먼트/위치 임베딩을 쓴다.

**참고할 선행 연구:**
- **Wu et al. (2016)** – WordPiece (서브워드 토크나이저). Google NMT 등.

---

## 7. 사전 훈련 데이터

**논문이 말하는 것:** BooksCorpus + 위키백과, **문서 단위** 코퍼스가 중요하다.

**참고할 선행 연구:**
- **Zhu et al. (2015)** – BooksCorpus.
- **Chelba et al. (2013)** – Billion Word Benchmark (문장 단위 셔플; BERT는 문서 단위와 대비).

---

## 8. 마스크드 LM (MLM)

**논문이 말하는 것:** 일부 토큰을 가리고 그걸 예측하는 **MLM**으로 양방향을 한 번에 학습한다.

**참고할 선행 연구:**
- **Taylor (1953)** – Cloze(빈칸 채우기) 작업. MLM이 여기서 영감을 받았다고 논문에서 밝힘.

---

## 9. 미세 조정 (문장 쌍, 입력·출력만 바꾸기)

**논문이 말하는 것:** 트랜스포머 셀프 어텐션 덕분에 문장 쌍도 한 시퀀스로 넣고, 입력·출력만 작업별로 바꿔서 끝까지 미세 조정한다.

**참고할 선행 연구:**
- **Parikh et al. (2016)**, **Seo et al. (2017)** – 문장 쌍을 **따로 인코딩한 뒤** 교차 어텐션을 쓰는 기존 패턴.

---

## 10. 실험 벤치마크 (GLUE, SQuAD, SWAG 등)

**참고할 선행 연구:**
- **Wang et al. (2018a)** – GLUE.
- **Rajpurkar et al. (2016)** – SQuAD v1.1.
- **Zellers et al. (2018)** – SWAG.

---

## 다운로드한 논문 위치

PDF는 `doc/BERT_Relative/papers/` 아래에 저장되어 있다. 다시 받으려면 프로젝트 루트에서:

```bash
bash doc/BERT_Relative/download_papers.sh
```

파일명은 아래 표의 **저장 파일명**을 따른다.

| 저장 파일명 | 논문 | 다운로드 소스 |
|-------------|------|----------------|
| Vaswani_etal_2017_Attention_Is_All_You_Need.pdf | Transformer | arXiv:1706.03762 |
| Peters_etal_2018_ELMo.pdf | ELMo | arXiv:1802.05365 |
| Radford_etal_2018_GPT.pdf | OpenAI GPT | OpenAI CDN |
| Dai_Le_2015_Semi_supervised_sequence_learning.pdf | Dai & Le | arXiv:1511.01432 |
| Howard_Ruder_2018_ULMFiT.pdf | ULMFiT | arXiv:1801.06146 |
| Wu_etal_2016_Google_NMT.pdf | WordPiece/Google NMT | arXiv:1609.08144 |
| Devlin_etal_2019_BERT.pdf | BERT | arXiv:1810.04805 |

Taylor (1953) Cloze, Zhu et al. (2015) BooksCorpus 등은 구독/도서관 또는 ACL/arXiv에서 별도 검색하여 이용하면 된다.
