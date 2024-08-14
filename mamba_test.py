import torch
import pytest
from mamba import MambaBlock, MambaLanguageModel

@pytest.fixture
def sample_input():
    return torch.randn(2, 10, 384)  # (batch_size, sequence_length, embedding_dim)

def test_mamba_block(sample_input):
    block = MambaBlock(384)
    output = block(sample_input)
    assert output.shape == sample_input.shape

def test_mamba_language_model():
    model = MambaLanguageModel(vocab_size=100)
    input_ids = torch.randint(0, 100, (2, 20))  # (batch_size, sequence_length)
    logits, loss = model(input_ids, targets=input_ids)
    assert logits.shape == (40, 100)
    assert loss is not None

def test_mamba_generate():
    model = MambaLanguageModel(vocab_size=100)
    input_ids = torch.randint(0, 100, (1, 10))  # (batch_size, initial_sequence_length)
    generated = model.generate(input_ids, max_new_tokens=5)
    assert generated.shape == (1, 15)  # Initial 10 + 5 new tokens

if __name__ == "__main__":
    pytest.main([__file__])