import torch
import pytest
from attention import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel

@pytest.fixture
def sample_input():
    return torch.randn(2, 10, 384)  # (batch_size, sequence_length, embedding_dim)

def test_head(sample_input):
    head = Head(64)
    output = head(sample_input)
    assert output.shape == (2, 10, 64)

def test_multihead_attention(sample_input):
    mha = MultiHeadAttention(6, 64)
    output = mha(sample_input)
    assert output.shape == sample_input.shape

def test_feedforward(sample_input):
    ff = FeedFoward(384)
    output = ff(sample_input)
    assert output.shape == sample_input.shape

def test_block(sample_input):
    block = Block(384, 6)
    output = block(sample_input)
    assert output.shape == sample_input.shape

def test_gpt_language_model():
    torch.set_default_device('cuda')
    model = GPTLanguageModel(vocab_size=100)
    input_ids = torch.randint(0, 100, (2, 20))  # (batch_size, sequence_length)
    logits, loss = model(input_ids, targets=input_ids)
    assert logits.shape == (40, 100)
    assert loss is not None

def test_gpt_generate():
    torch.set_default_device('cuda')
    model = GPTLanguageModel(vocab_size=100)
    input_ids = torch.randint(0, 100, (1, 10))  # (batch_size, initial_sequence_length)
    generated = model.generate(input_ids, max_new_tokens=5)
    assert generated.shape == (1, 15)  # Initial 10 + 5 new tokens

if __name__ == "__main__":
    pytest.main([__file__])