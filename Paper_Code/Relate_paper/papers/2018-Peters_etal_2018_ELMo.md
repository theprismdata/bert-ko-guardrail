# Deep Contextualized Word Representations (딥 문맥화 단어 표현, ELMo)

**원문:** Peters et al., 2018 — *Deep contextualized word representations*  
**저자:** Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner (Allen Institute for AI), Christopher Clark, Kenton Lee, Luke Zettlemoyer (University of Washington)  
**arXiv:** 1802.05365v2 [cs.CL] (22 Mar 2018)

---

## 읽기 전에: 핵심 개념

**ELMo (Embeddings from Language Models)**  
단어마다 문맥에 따라 다른 벡터를 주는 표현을, **양방향 언어 모델(biLM)**의 내부 상태를 이용해 만드는 방법이다. 기존 GloVe·Word2Vec 같은 고정 임베딩과 달리, 같은 단어라도 문장마다 다른 벡터를 갖는다. 다의어·문법 정보를 잘 반영하고, 다양한 NLP 태스크의 입력으로 그대로 붙여 쓸 수 있다.

**biLM (bidirectional Language Model)**  
앞쪽 문맥만 쓰는 순방향 LM과 뒤쪽 문맥만 쓰는 역방향 LM을 함께 학습한 모델이다. 각 토큰에 대해 두 방향의 LSTM 출력을 합쳐(concatenate) 문맥 표현을 만든다.

---

## Abstract (초록)

**원문**  
We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.

**번역**  
본 논문은 (1) 단어 사용의 복잡한 특성(문법, 의미)과 (2) 언어적 문맥에 따른 사용의 변화(다의어 모델링)를 모두 반영하는 새로운 종류의 **딥 문맥화 단어 표현**을 소개한다. 이 단어 벡터는 대규모 텍스트 코퍼스로 사전 학습된 **딥 양방향 언어 모델(biLM)**의 내부 상태에 대한 학습된 함수이다. 이러한 표현을 기존 모델에 쉽게 추가할 수 있으며, 질의응답·텍스트 함의·감성 분석을 포함한 여섯 가지 난이도 높은 NLP 과제에서 최신 성능을 크게 향상시킨다. 또한 사전 학습된 네트워크의 **깊은 내부 표현을 노출하는 것**이 중요함을 분석을 통해 보이며, 하류 태스크 모델이 서로 다른 종류의 준지도 신호를 섞어 쓸 수 있게 한다.

---

## 1 Introduction (서론)

**원문**  
Pre-trained word representations (Mikolov et al., 2013; Pennington et al., 2014) are a key component in many neural language understanding models. However, learning high quality representations can be challenging. They should ideally model both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). In this paper, we introduce a new type of deep contextualized word representation that directly addresses both challenges, can be easily integrated into existing models, and significantly improves the state of the art in every considered case across a range of challenging language understanding problems.

Our representations differ from traditional word type embeddings in that each token is assigned a representation that is a function of the entire input sentence. We use vectors derived from a bidirectional LSTM that is trained with a coupled language model (LM) objective on a large text corpus. For this reason, we call them ELMo (Embeddings from Language Models) representations. Unlike previous approaches for learning contextualized word vectors (Peters et al., 2017; McCann et al., 2017), ELMo representations are deep, in the sense that they are a function of all of the internal layers of the biLM. More specifically, we learn a linear combination of the vectors stacked above each input word for each end task, which markedly improves performance over just using the top LSTM layer.

**번역**  
사전 학습된 단어 표현(Mikolov et al., 2013; Pennington et al., 2014)은 많은 신경망 언어 이해 모델에서 핵심 요소이다. 그러나 고품질 표현을 학습하는 것은 쉽지 않다. 이상적으로는 (1) 단어 사용의 복잡한 특성(문법, 의미)과 (2) 언어적 문맥에 따른 사용의 변화(다의어 모델링)를 모두 반영해야 한다. 본 논문에서는 이 두 가지를 직접 다루는 새로운 **딥 문맥화 단어 표현**을 소개한다. 이 표현은 기존 모델에 쉽게 통합할 수 있으며, 다양한 난이도 높은 언어 이해 과제에서 고려한 모든 경우에 최신 성능을 크게 향상시킨다.

우리 표현은 전통적인 단어 유형 임베딩과 달리, **각 토큰에 전체 입력 문장의 함수인 표현**이 할당된다. 대규모 텍스트 코퍼스에서 결합된 언어 모델(LM) 목적 함수로 학습된 **양방향 LSTM**에서 나온 벡터를 사용하며, 이를 **ELMo(Embeddings from Language Models)** 표현이라 부른다. 기존 문맥화 단어 벡터 학습(Peters et al., 2017; McCann et al., 2017)과 달리 ELMo는 biLM의 **모든 내부 레이어**의 함수라는 의미에서 **딥**하다. 구체적으로 각 종단 태스크마다 각 입력 단어 위에 쌓인 벡터들의 **선형 결합**을 학습하며, 최상위 LSTM 레이어만 쓰는 것보다 성능이 뚜렷히 좋아진다.

**원문**  
Combining the internal states in this manner allows for very rich word representations. Using intrinsic evaluations, we show that the higher-level LSTM states capture context-dependent aspects of word meaning (e.g., they can be used without modification to perform well on supervised word sense disambiguation tasks) while lower-level states model aspects of syntax (e.g., they can be used to do part-of-speech tagging). Simultaneously exposing all of these signals is highly beneficial, allowing the learned models select the types of semi-supervision that are most useful for each end task.

Extensive experiments demonstrate that ELMo representations work extremely well in practice. We first show that they can be easily added to existing models for six diverse and challenging language understanding problems, including textual entailment, question answering and sentiment analysis. The addition of ELMo representations alone significantly improves the state of the art in every case, including up to 20% relative error reductions. For tasks where direct comparisons are possible, ELMo outperforms CoVe (McCann et al., 2017), which computes contextualized representations using a neural machine translation encoder. Finally, an analysis of both ELMo and CoVe reveals that deep representations outperform those derived from just the top layer of an LSTM.

**번역**  
이렇게 내부 상태를 결합하면 매우 풍부한 단어 표현이 가능해진다. 내재적 평가를 통해, 상위 LSTM 상태는 문맥에 따른 단어 의미(지도 학습 단어 의미 중의성 해소 등)를 포착하고, 하위 상태는 문법(품사 태깅 등)을 모델링함을 보인다. 이러한 신호를 **동시에 모두 노출**하는 것이 매우 유리하며, 학습된 모델이 각 종단 태스크에 가장 유용한 준지도 유형을 선택할 수 있게 한다.

광범위한 실험에서 ELMo 표현이 실제로 매우 잘 동작함을 보인다. 텍스트 함의·질의응답·감성 분석을 포함한 여섯 가지 다양한 언어 이해 과제에 기존 모델에 쉽게 추가할 수 있음을 먼저 보이고, ELMo만 추가해도 모든 경우에 최신 성능을 크게 개선하며 상대 오류 감소가 최대 약 20%에 이른다. 직접 비교가 가능한 태스크에서는 신경망 기계 번역 인코더로 문맥화 표현을 계산하는 CoVe(McCann et al., 2017)보다 ELMo가 우수하다. 마지막으로 ELMo와 CoVe에 대한 분석에서, 딥 표현이 LSTM 최상위 레이어만 쓴 표현보다 성능이 더 좋음을 보인다.

---

## 2 Related Work (관련 연구)

**원문**  
Due to their ability to capture syntactic and semantic information of words from large scale unlabeled text, pretrained word vectors (Turian et al., 2010; Mikolov et al., 2013; Pennington et al., 2014) are a standard component of most state-of-the-art NLP architectures, including for question answering (Liu et al., 2017), textual entailment (Chen et al., 2017) and semantic role labeling (He et al., 2017). However, these approaches for learning word vectors only allow a single context-independent representation for each word.

Previously proposed methods overcome some of the shortcomings of traditional word vectors by either enriching them with subword information (e.g., Wieting et al., 2016; Bojanowski et al., 2017) or learning separate vectors for each word sense (e.g., Neelakantan et al., 2014). Our approach also benefits from subword units through the use of character convolutions, and we seamlessly incorporate multi-sense information into downstream tasks without explicitly training to predict predefined sense classes.

**번역**  
대규모 비레이블 텍스트에서 단어의 문법·의미 정보를 포착할 수 있어, 사전 학습된 단어 벡터(Turian et al., 2010; Mikolov et al., 2013; Pennington et al., 2014)는 질의응답(Liu et al., 2017), 텍스트 함의(Chen et al., 2017), 의미역 라벨링(He et al., 2017) 등 대부분의 최신 NLP 구조에서 표준 구성 요소이다. 그러나 이런 단어 벡터 학습 방식은 단어당 **문맥에 무관한 하나의 표현**만 허용한다.

기존 제안 방법들은 서브워드 정보 보강(Wieting et al., 2016; Bojanowski et al., 2017)이나 단어 의미별 별도 벡터 학습(Neelakantan et al., 2014) 등으로 전통적 단어 벡터의 한계를 일부 극복한다. 본 접근은 문자 합성곱을 통해 서브워드 단위의 이점을 얻고, 미리 정의된 의미 클래스를 예측하도록 명시적으로 학습하지 않고도 다의 정보를 하류 태스크에 자연스럽게 통합한다.

**원문**  
Other recent work has also focused on learning context-dependent representations. context2vec (Melamud et al., 2016) uses a bidirectional LSTM to encode the context around a pivot word. Other approaches for learning contextual embeddings include the pivot word itself in the representation and are computed with the encoder of either a supervised neural machine translation (MT) system (CoVe; McCann et al., 2017) or an unsupervised language model (Peters et al., 2017). Both of these approaches benefit from large datasets, although the MT approach is limited by the size of parallel corpora. In this paper, we take full advantage of access to plentiful monolingual data, and train our biLM on a corpus with approximately 30 million sentences (Chelba et al., 2014). We also generalize these approaches to deep contextual representations, which we show work well across a broad range of diverse NLP tasks.

**번역**  
다른 최근 연구는 문맥 의존 표현 학습에 초점을 맞춘다. context2vec(Melamud et al., 2016)은 중심 단어 주변 문맥을 인코딩하기 위해 양방향 LSTM을 사용한다. 문맥 임베딩을 학습하는 다른 접근들은 표현에 중심 단어 자체를 포함하며, 지도 학습 신경망 기계 번역(MT) 인코더(CoVe; McCann et al., 2017) 또는 비지도 언어 모델(Peters et al., 2017)의 인코더로 계산된다. 두 접근 모두 대규모 데이터의 이점을 받지만, MT 접근은 병렬 코퍼스 크기에 제한된다. 본 논문에서는 풍부한 단일 언어 데이터를 활용해 약 3천만 문장 코퍼스(Chelba et al., 2014)에서 biLM을 학습하고, 이러한 접근을 **딥 문맥 표현**으로 일반화하여 다양한 NLP 태스크 전반에서 잘 동작함을 보인다.

**원문**  
Previous work has also shown that different layers of deep biRNNs encode different types of information. For example, introducing multi-task syntactic supervision (e.g., part-of-speech tags) at the lower levels of a deep LSTM can improve overall performance of higher level tasks such as dependency parsing (Hashimoto et al., 2017) or CCG super tagging (Søgaard and Goldberg, 2016). In an RNN-based encoder-decoder machine translation system, Belinkov et al. (2017) showed that the representations learned at the first layer in a 2-layer LSTM encoder are better at predicting POS tags then second layer. Finally, the top layer of an LSTM for encoding word context (Melamud et al., 2016) has been shown to learn representations of word sense. We show that similar signals are also induced by the modified language model objective of our ELMo representations, and it can be very beneficial to learn models for downstream tasks that mix these different types of semi-supervision.

**번역**  
선행 연구에서는 딥 양방향 RNN의 서로 다른 레이어가 서로 다른 종류의 정보를 인코딩함이 이미 보여졌다. 예를 들어 딥 LSTM의 하위 레이어에 다중 과제 문법 감독(품사 태그 등)을 넣으면 의존 구문 분석(Hashimoto et al., 2017), CCG 슈퍼 태깅(Søgaard and Goldberg, 2016) 같은 상위 과제의 전체 성능이 향상될 수 있다. RNN 기반 인코더-디코더 기계 번역에서 Belinkov et al. (2017)은 2층 LSTM 인코더의 첫 번째 레이어 표현이 두 번째 레이어보다 품사 태그 예측에 더 낫다고 했다. 단어 문맥 인코딩용 LSTM의 최상위 레이어(Melamud et al., 2016)는 단어 의미 표현을 학습하는 것으로 알려져 있다. 본 논문에서는 ELMo 표현의 수정된 언어 모델 목적 함수에 의해 유사한 신호가 유도됨을 보이고, 이러한 서로 다른 준지도 유형을 섞는 하류 태스크용 모델을 학습하는 것이 매우 유리할 수 있음을 보인다.

**원문**  
Dai and Le (2015) and Ramachandran et al. (2017) pretrain encoder-decoder pairs using language models and sequence autoencoders and then fine tune with task specific supervision. In contrast, after pretraining the biLM with unlabeled data, we fix the weights and add additional task-specific model capacity, allowing us to leverage large, rich and universal biLM representations for cases where downstream training data size dictates a smaller supervised model.

**번역**  
Dai and Le (2015)와 Ramachandran et al. (2017)은 언어 모델과 시퀀스 오토인코더로 인코더-디코더 쌍을 사전 학습한 뒤 태스크별 지도 학습으로 미세 조정한다. 이와 달리 우리는 비레이블 데이터로 biLM을 사전 학습한 후 **가중치를 고정**하고 태스크별 추가 모델 용량만 더해, 하류 학습 데이터 크기가 작은 지도 모델을 요구하는 경우에도 크고 풍부한 범용 biLM 표현을 활용할 수 있게 한다.

---

## 3 ELMo: Embeddings from Language Models

**원문**  
Unlike most widely used word embeddings (Pennington et al., 2014), ELMo word representations are functions of the entire input sentence, as described in this section. They are computed on top of two-layer biLMs with character convolutions (Sec. 3.1), as a linear function of the internal network states (Sec. 3.2). This setup allows us to do semi-supervised learning, where the biLM is pretrained at a large scale (Sec. 3.4) and easily incorporated into a wide range of existing neural NLP architectures (Sec. 3.3).

**번역**  
가장 널리 쓰이는 단어 임베딩(Pennington et al., 2014)과 달리, ELMo 단어 표현은 이 절에서 설명하듯 **전체 입력 문장의 함수**이다. 문자 합성곱을 쓰는 2층 biLM 위에서(Sec. 3.1), 내부 네트워크 상태의 선형 함수(Sec. 3.2)로 계산된다. 이 구성을 통해 biLM을 대규모로 사전 학습(Sec. 3.4)하고 다양한 기존 신경 NLP 구조에 쉽게 통합(Sec. 3.3)하는 준지도 학습이 가능하다.

### 3.1 Bidirectional language models (양방향 언어 모델)

**원문**  
Given a sequence of N tokens, (t1, t2, ..., tN), a forward language model computes the probability of the sequence by modeling the probability of token tk given the history (t1, ..., tk−1):

$$p(t_1, t_2, \ldots, t_N) = \prod_{k=1}^{N} p(t_k \mid t_1, t_2, \ldots, t_{k-1}).$$

Recent state-of-the-art neural language models compute a context-independent token representation x_LM_k (via token embeddings or a CNN over characters) then pass it through L layers of forward LSTMs. At each position k, each LSTM layer outputs a context-dependent representation h_LM_k,j (forward, j = 1,...,L). The top layer LSTM output is used to predict the next token tk+1 with a Softmax layer.

A backward LM is similar to a forward LM, except it runs over the sequence in reverse, predicting the previous token given the future context:

$$p(t_1, t_2, \ldots, t_N) = \prod_{k=1}^{N} p(t_k \mid t_{k+1}, t_{k+2}, \ldots, t_N).$$

Each backward LSTM layer j produces representations of tk given (tk+1, ..., tN). A biLM combines both a forward and backward LM. The formulation jointly maximizes the log likelihood of the forward and backward directions; the parameters for the token representation and Softmax layer are tied in both directions, while the LSTMs in each direction have separate parameters.

**번역**  
N개 토큰 시퀀스 (t1, t2, ..., tN)에 대해, **순방향 언어 모델**은 이력 (t1, ..., tk−1)이 주어졌을 때 tk의 확률을 모델링해 시퀀스 확률을 계산한다(위 식). 최신 신경 언어 모델은 문맥 무관 토큰 표현 x_LM_k(토큰 임베딩 또는 문자 CNN)를 계산한 뒤 L층 순방향 LSTM을 통과시킨다. 각 위치 k에서 각 LSTM 레이어는 문맥 의존 표현 h_LM_k,j(순방향, j=1,...,L)를 출력하고, 최상위 LSTM 출력으로 Softmax를 써 다음 토큰 tk+1을 예측한다.

**역방향 LM**은 시퀀스를 역순으로 돌며 미래 문맥이 주어졌을 때 이전 토큰을 예측한다(위 역방향 식). 각 역방향 LSTM 레이어 j는 (tk+1, ..., tN)이 주어졌을 때 tk의 표현을 만든다. **biLM**은 순방향·역방향 LM을 결합하며, 두 방향의 로그 우도를 함께 최대화한다. 토큰 표현과 Softmax 파라미터는 두 방향에서 묶고, 각 방향의 LSTM 파라미터는 따로 둔다.

### 3.2 ELMo

**원문**  
ELMo is a task specific combination of the intermediate layer representations in the biLM. For each token tk, a L-layer biLM computes a set of 2L+1 representations

$$R_k = \{ x^{LM}_k,\; \overrightarrow{h}^{LM}_{k,j},\; \overleftarrow{h}^{LM}_{k,j} \mid j=1,\ldots,L \} = \{ h^{LM}_{k,j} \mid j=0,\ldots,L \},$$

where h_LM_k,0 is the token layer and h_LM_k,j = [forward; backward] for each biLSTM layer.

For inclusion in a downstream model, ELMo collapses all layers in R into a single vector, ELMo_k = E(R_k; Theta_e). In the simplest case, ELMo just selects the top layer. More generally, a task specific weighting of all biLM layers is computed:

$$ELMo^{task}_k = \gamma^{task} \sum_{j=0}^{L} s^{task}_j h^{LM}_{k,j}.$$

Here s_task are softmax-normalized weights and the scalar gamma_task lets the task model scale the entire ELMo vector. Layer normalization (Ba et al., 2016) is sometimes applied to each biLM layer before weighting.

**번역**  
ELMo는 biLM의 **중간 레이어 표현들의 태스크별 결합**이다. 각 토큰 tk에 대해 L층 biLM은 2L+1개 표현 집합 R_k를 계산한다(위 식). h_LM_k,0은 토큰 레이어이고, 각 biLSTM 레이어에 대해 h_LM_k,j는 [순방향; 역방향] 결합이다.

하류 모델에 넣기 위해 ELMo는 R의 모든 레이어를 하나의 벡터 ELMo_k = E(R_k; Theta_e)로 합친다. 가장 단순한 경우 최상위 레이어만 선택한다. 일반적으로는 모든 biLM 레이어에 대한 태스크별 가중 합(위 식)을 계산한다. s_task는 softmax 정규화 가중치이고, 스칼라 gamma_task로 태스크 모델이 전체 ELMo 벡터의 스케일을 조절한다. 필요 시 가중치 적용 전에 각 biLM 레이어에 층 정규화(Ba et al., 2016)를 적용한다.

### 3.3 Using biLMs for supervised NLP tasks (지도 NLP 태스크에서 biLM 사용)

**원문**  
Given a pre-trained biLM and a supervised architecture for a target NLP task, it is a simple process to use the biLM to improve the task model. We run the biLM and record all layer representations for each word, then let the end task model learn a linear combination of these representations.

Most supervised NLP models share a common architecture at the lowest layers: they form a context-independent token representation xk (e.g. pretrained word embeddings and optionally character-based representations), then a context-sensitive representation hk via bidirectional RNNs, CNNs, or feed forward networks. To add ELMo, we freeze the biLM weights and concatenate ELMo_k with xk, passing [xk; ELMo_k] into the task RNN. For some tasks (e.g. SNLI, SQuAD), further improvements are seen by also including ELMo at the output of the task RNN (another set of output-specific linear weights, replacing hk with [hk; ELMo_k]). It is also beneficial to add moderate dropout to ELMo and in some cases to regularize the ELMo weights with lambda ||w||^2_2 so they stay close to an average of all biLM layers.

**번역**  
사전 학습된 biLM과 목표 NLP 태스크용 지도 구조가 주어지면, biLM으로 태스크 모델을 개선하는 절차는 단순하다. biLM을 실행해 각 단어에 대한 모든 레이어 표현을 기록한 뒤, 종단 태스크 모델이 이 표현들의 선형 결합을 학습하게 한다.

대부분의 지도 NLP 모델은 최하위 레이어에서 공통 구조를 공유한다. 문맥 무관 토큰 표현 xk(사전 학습 단어 임베딩, 선택적으로 문자 기반 표현)를 만든 뒤, 양방향 RNN·CNN·전방향 네트워크로 문맥 민감 표현 hk를 만든다. ELMo를 추가할 때는 biLM 가중치를 고정하고 ELMo_k를 xk와 결합해 [xk; ELMo_k]를 태스크 RNN에 넣는다. SNLI, SQuAD 등 일부 태스크에서는 태스크 RNN의 출력에도 ELMo를 넣어(hk를 [hk; ELMo_k]로 대체) 추가 개선이 있다. ELMo에 적당한 dropout을 주고, 경우에 따라 lambda ||w||^2_2로 ELMo 가중치를 정규화해 모든 biLM 레이어의 평균에 가깝게 두는 것도 유리하다.

### 3.4 Pre-trained bidirectional language model architecture (사전 학습 biLM 구조)

**원문**  
The pre-trained biLMs are similar to the architectures in Jozefowicz et al. (2016) and Kim et al. (2015), but modified to support joint training of both directions and to add a residual connection between LSTM layers. To balance perplexity with model size and compute, embedding and hidden dimensions are halved from the single best CNN-BIG-LSTM in Jozefowicz et al. (2016). The final model uses L=2 biLSTM layers with 4096 units and 512-dim projections and a residual connection from the first to second layer. The context insensitive type representation uses 2048 character n-gram convolutional filters, two highway layers, and a linear projection to 512 dims. Thus the biLM provides three layers of representations for each input token (including OOV due to character-based input). After training for 10 epochs on the 1B Word Benchmark (Chelba et al., 2014), the average forward and backward perplexity is 39.7. Fine tuning the biLM on domain-specific data often reduces perplexity and improves downstream performance (domain transfer).

**번역**  
사전 학습된 biLM은 Jozefowicz et al. (2016), Kim et al. (2015)의 구조와 비슷하지만, 두 방향의 공동 학습과 LSTM 레이어 간 잔차 연결을 위해 수정되었다. Perplexity와 모델 크기·연산의 균형을 위해 Jozefowicz et al. (2016)의 CNN-BIG-LSTM 대비 임베딩·은닉 차원을 절반으로 줄였다. 최종 모델은 4096 유닛·512차원 투영의 2층 biLSTM과 1층에서 2층으로의 잔차 연결을 사용한다. 문맥 무관 유형 표현은 2048개 문자 n-gram 합성곱 필터, 2개의 highway 레이어, 512차원 선형 투영을 사용한다. 따라서 biLM은 각 입력 토큰에 대해 3층 표현을 제공하며(문자 기반 입력으로 OOV 포함). 1B Word Benchmark(Chelba et al., 2014)로 10 에폭 학습 후 평균 순·역 perplexity는 39.7이다. 도메인별 데이터로 biLM을 미세 조정하면 perplexity가 떨어지고 하류 성능이 올라가는 경우가 많다(도메인 전이).

---

## 4 Evaluation (평가)

**원문**  
Table 1 shows the performance of ELMo across six benchmark NLP tasks. In every task, simply adding ELMo establishes a new state-of-the-art result, with relative error reductions ranging from 6–20% over strong baselines. Below we summarize the task-level results.

**번역**  
표 1은 여섯 가지 벤치마크 NLP 태스크에서의 ELMo 성능을 보여 준다. 모든 태스크에서 ELMo만 추가해도 강한 베이스라인 대비 상대 오류 6–20% 감소로 새로운 최신 결과를 달성했다. 아래에 태스크별 결과를 요약한다.

| TASK | PREVIOUS SOTA | OUR BASELINE | ELMo + BASELINE | INCREASE (ABS/REL) |
|------|---------------|--------------|-----------------|--------------------|
| SQuAD | Liu et al. 84.4 | 81.1 | 85.8 | 4.7 / 24.9% |
| SNLI | Chen et al. 88.6 | 88.0 | 88.7 | 0.7 / 5.8% |
| SRL | He et al. 81.7 | 81.4 | 84.6 | 3.2 / 17.2% |
| Coref | Lee et al. 67.2 | 67.2 | 70.4 | 3.2 / 9.8% |
| NER | Peters et al. 91.93 | 90.15 | 92.22 | 2.06 / 21% |
| SST-5 | McCann et al. 53.7 | 51.4 | 54.7 | 3.3 / 6.8% |

- **Question answering (SQuAD):** 베이스라인(Clark and Gardner, 2017, BiDAF 개선)에 ELMo를 넣어 테스트 F1이 81.1%에서 85.8%로 4.7%p 상승(상대 오류 24.9% 감소). CoVe 추가 대비 개선 폭이 훨씬 크다.
- **Textual entailment (SNLI):** ESIM(Chen et al., 2017)에 ELMo 추가 시 평균 0.7% 정확도 향상. 5개 앙상블로 89.3%로 이전 최고 앙상블 88.9% 초과.
- **Semantic role labeling (SRL):** He et al. (2017) 재구현에 ELMo 추가 시 단일 모델 F1이 81.4%에서 84.6%로 3.2%p 상승, OntoNotes 벤치마크 신규 SOTA.
- **Coreference resolution:** Lee et al. (2017) 엔드투엔드 스팬 기반 모델에 ELMo 추가 시 CoNLL 2012 기준 평균 F1이 67.2에서 70.4로 3.2%p 상승, 신규 SOTA.
- **Named entity recognition (CoNLL 2003 NER):** biLSTM-CRF 베이스라인에 ELMo 추가, 5회 실행 평균 F1 92.22%. Peters et al. (2017)과의 차이는 태스크 모델이 **모든 biLM 레이어의 가중 평균**을 학습한 점이며, 최상위 레이어만 쓰는 것보다 여러 태스크에서 성능이 좋다.
- **Sentiment analysis (SST-5):** McCann et al. (2017) BCN에 CoVe 대신 ELMo를 넣어 SOTA 대비 1.0%p 절대 정확도 향상.

---

## 5 Analysis (분석)

### 5.1 Alternate layer weighting schemes (레이어 가중 방식)

**원문**  
Alternatives to combining biLM layers include using only the last layer (as in TagLM, CoVe) or learning weights with different regularization (lambda). Large lambda (e.g. 1) effectively gives a simple average over layers; small lambda (e.g. 0.001) allows layer weights to vary. Table 2 (in paper) shows that using all layers improves over using only the last layer, and learning per-layer weights (small lambda) improves further. The same trend holds for CoVe but with smaller gains.

**번역**  
biLM 레이어 결합 방식으로는 최상위 레이어만 사용(TagLM, CoVe)하거나, 정규화 강도 lambda를 바꿔 가중치를 학습하는 방법이 있다. lambda가 크면(예: 1) 사실상 레이어 단순 평균, 작으면(예: 0.001) 레이어별 가중치가 자유롭게 학습된다. 논문 표 2에서 모든 레이어를 쓰는 것이 최상위만 쓰는 것보다 좋고, 레이어별 가중치 학습(작은 lambda)이 추가로 유리함이 나온다. CoVe에서도 같은 경향이지만 개선 폭은 작다.

### 5.2 Where to include ELMo? (ELMo를 어디에 넣을지)

**원문**  
Including ELMo at both the input and output of the task biRNN helps for SNLI and SQuAD (which use attention after the biRNN), but for SRL and coreference the best is to include ELMo only at the input. So for architectures with attention after the RNN, adding ELMo at the output lets the model attend directly to biLM representations; for SRL, task-specific context may matter more than biLM context.

**번역**  
태스크 biRNN의 입력과 출력 모두에 ELMo를 넣으면 SNLI와 SQuAD( biRNN 뒤에 attention 사용)에서 도움이 되지만, SRL과 공동참조에서는 입력에만 넣는 것이 최선이다. RNN 뒤에 attention이 있는 구조에서는 출력에 ELMo를 두면 모델이 biLM 표현에 직접 attention할 수 있고, SRL에서는 태스크별 문맥이 biLM 문맥보다 더 중요할 수 있다.

### 5.3 What information is captured by the biLM? (biLM이 포착하는 정보)

**원문**  
The biLM must encode information useful for NLP beyond static word vectors, including disambiguation of word meaning by context. For example, GloVe neighbors of "play" mix POS and senses; biLM context representations of "play" in SemCor give neighbors that match both POS and sense. Intrinsic evaluation: (1) **Word sense disambiguation (WSD):** biLM top-layer representations achieve F1 69.0 (fine-grained WSD), competitive with supervised WSD systems; CoVe layers follow a similar pattern but biLM outperforms CoVe. (2) **POS tagging:** A linear classifier on biLM representations reaches accuracies competitive with task-specific biLSTMs; the **first** biLM layer is better than the top layer for POS, consistent with syntax being captured at lower layers. So higher layers capture context-dependent meaning (WSD), lower layers syntax (POS); exposing all layers lets downstream models mix these signals. biLM representations also transfer better to WSD and POS than CoVe, consistent with ELMo beating CoVe in downstream tasks.

**번역**  
biLM은 정적 단어 벡터로는 얻기 어려운, 문맥에 의한 단어 의미 중의성 해소 등 NLP에 유용한 정보를 인코딩해야 한다. 예를 들어 "play"의 GloVe 이웃은 품사와 의미가 섞여 있지만, SemCor 문장에서 "play"에 대한 biLM 문맥 표현의 이웃은 품사와 의미가 모두 맞는다. 내재적 평가: (1) **단어 의미 중의성 해소(WSD):** biLM 최상위 레이어 표현이 세밀 WSD에서 F1 69.0으로 지도 WSD 시스템과 비슷한 수준; CoVe도 유사한 층별 패턴이지만 biLM이 더 우수. (2) **품사 태깅:** biLM 표현에 선형 분류기를 붙이면 태스크별 biLSTM과 비슷한 정확도; **첫 번째** biLM 레이어가 최상위보다 품사에 유리해, 하위 레이어가 문법을 담당한다는 기존 결과와 일치. 따라서 상위 레이어는 문맥 의존 의미(WSD), 하위 레이어는 문법(POS)을 담당하고, 모든 레이어를 노출하면 하류 모델이 이 신호들을 섞어 쓸 수 있다. biLM 표현은 CoVe보다 WSD·POS로의 전이도 좋아, 하류 태스크에서 ELMo가 CoVe를 앞서는 것과 맞는다.

### 5.4 Sample efficiency (샘플 효율)

**원문**  
ELMo increases sample efficiency: e.g. SRL without ELMo needs 486 epochs to reach max dev F1; with ELMo the baseline max is exceeded by epoch 10. When varying training set size from 0.1% to 100%, improvements with ELMo are largest for smaller sets; e.g. for SRL, ELMo with 1% of the data is about as good as the baseline with 10%.

**번역**  
ELMo는 샘플 효율을 높인다. 예: ELMo 없이 SRL은 최대 dev F1에 486 에폭이 필요하고, ELMo를 쓰면 10 에폭에 베이스라인 최대를 넘긴다. 훈련 집합 크기를 0.1%에서 100%까지 바꿀 때, ELMo의 개선은 작은 데이터일수록 크다. SRL의 경우 데이터 1%로 ELMo를 쓴 모델이 데이터 10% 베이스라인과 비슷한 수준이다.

### 5.5 Visualization of learned weights (학습된 가중치 시각화)

**원문**  
Softmax-normalized layer weights: at the input layer, the task model often favors the first biLSTM layer (strongly for coref and SQuAD). Output layer weights are more balanced with a slight preference for lower layers.

**번역**  
Softmax 정규화된 레이어 가중치: 입력 레이어에서는 태스크 모델이 첫 번째 biLSTM 레이어를 선호하는 경우가 많고(공동참조·SQuAD에서 특히 그렇다). 출력 레이어 가중치에는 하위 레이어에 대한 약한 선호가 있지만 전반적으로 더 균형 잡혀 있다.

---

## 6 Conclusion (결론)

**원문**  
We have introduced a general approach for learning high-quality deep context-dependent representations from biLMs, and shown large improvements when applying ELMo to a broad range of NLP tasks. Through ablations and controlled experiments, we confirmed that biLM layers encode different types of syntactic and semantic information and that using all layers improves overall task performance.

**번역**  
biLM으로부터 고품질의 딥 문맥 의존 표현을 학습하는 일반적인 방법을 제안했고, ELMo를 다양한 NLP 태스크에 적용했을 때 큰 개선을 보였다. Ablation 및 통제 실험을 통해 biLM 레이어가 서로 다른 종류의 문법·의미 정보를 인코딩하며, 모든 레이어를 사용하는 것이 전반적인 태스크 성능을 높인다는 것을 확인했다.

---

## References (참고 문헌)

논문 본문 및 부록에 인용된 주요 문헌: Ba et al. 2016 (Layer normalization), Belinkov et al. 2017 (MT encoder layers), Bojanowski et al. 2017 (subword), Bowman et al. 2015 (SNLI), Chelba et al. 2014 (1B Word Benchmark), Chen et al. 2017 (ESIM), Clark & Gardner 2017 (SQuAD), Dai & Le 2015 (semi-supervised sequence learning), He et al. 2017 (SRL), Lee et al. 2017 (coreference), McCann et al. 2017 (CoVe), Melamud et al. 2016 (context2vec), Mikolov et al. 2013, Pennington et al. 2014 (GloVe), Peters et al. 2017 (TagLM, biLM), Ramachandran et al. 2017, Jozefowicz et al. 2016 (CNN-BIG-LSTM), Kim et al. 2015 (character-aware LM), 기타 논문은 원문 참조.

---

## Supplemental Material 요약 (부록)

- **A.1 Fine tuning biLM:** 태스크별 데이터로 biLM을 1 에폭 미세 조정하면 대부분 태스크에서 perplexity가 크게 감소하고(예: SNLI 72.1 → 16.8), SNLI 등에서는 하류 정확도도 소폭 상승. 감성 분류는 미세 조정 유무에 따라 비슷.
- **A.2 Importance of gamma in Eq. (1):** biLM 내부 표현과 태스크 표현의 분포 차이 때문에 스케일 파라미터 gamma가 최적화에 중요. 없으면 "last only" 설정에서 SNLI는 베이스라인보다 낮고, SRL은 학습이 실패하기도 함.
- **A.3–A.8** 각 태스크(SNLI, SQuAD, SRL, Coref, NER, SST)에 대한 베이스라인 구조, 하이퍼파라미터, ELMo 추가 방식(입력/출력, lambda, layer norm 등) 및 테스트 결과 표가 원문 부록에 수록되어 있음.
