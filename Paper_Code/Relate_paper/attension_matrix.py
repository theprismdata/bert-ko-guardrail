"""
BPE·WordPiece 토크나이저 예제
- 문장 → 토크나이징 → ID → 임베딩
- KLUE BERT(WordPiece) 토크나이저 사용
"""

import numpy as np
from transformers import AutoTokenizer

# 1) 문장 -> 토크나이저 (WordPiece/BPE) -> 토큰 & ID
# 한국어 문장이므로 KLUE BERT(WordPiece) 사용. BERT 계열은 WordPiece, GPT 계열은 BPE를 주로 사용
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
text = "최근 사드 배치 때문에 한중 관계가 악화되면서 많은 중국인들이 불안해하고 있다."

# 토크나이징: 문자열 -> 서브워드 토큰 열 -> 정수 ID 열
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
# 예시에서는 앞 3개만 사용
tokens = tokens[:3]
print(tokens)
input_ids = np.array(input_ids[:3])
print(input_ids)

print("토큰:", tokens)
print("ID:", input_ids.tolist())
# (문서 표의 1024, 2047, 3512는 예시용 시뮬레이션 값이며, 실제 BPE/WordPiece ID는 모델마다 다름)

# 2) 임베딩 행렬 (실제 BERT는 768차원 등; 여기서는 예시로 3차원)
vocab_size = tokenizer.vocab_size
d_model = 3
emb_table = np.random.randn(vocab_size, d_model) * 0.02
# 앞 3개 토큰에 해당하는 행만 단위벡터로 둠 (출력 형태를 문서 표와 비슷하게)
for i, idx in enumerate(input_ids):
    v = np.zeros(d_model)
    v[i] = 1.0
    emb_table[idx] = v

# 3) ID -> 행 조회 -> 벡터 (임베딩 층 출력)
X = emb_table[input_ids]
print("임베딩 출력 X:\n", X)
