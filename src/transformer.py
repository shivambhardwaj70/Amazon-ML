# transformer.py (skeleton -- finalize once you know the actual task)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "microsoft/deberta-v3-base"  # base first; large only if GPU/time allow

# Tips:
# - max_length match data (256 typical, 512 if long)
# - fp16/bf16 = True
# - LR 1e-5 to 2e-5; 2-3 epochs; warmup 10%; weight_decay 0.01
# - save OOF logits per fold to stack with LGBM
# - num_labels=1 + MSE loss for regression tasks
