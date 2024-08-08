import torch
import pytest
from xlstm import xLSTM, XLSTMLanguageModel, mLSTMblock, sLSTMblock

@pytest.fixture
def sample_input():
    return torch.randn(32, 128, 256)  # (batch_size, sequence_length, embedding_dim)

def test_xLSTM(sample_input):
    layers = ['s', 'm', 's']
    model = xLSTM(layers, sample_input)
    
    # Test initialization
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], sLSTMblock)
    assert isinstance(model.layers[1], mLSTMblock)
    assert isinstance(model.layers[2], sLSTMblock)
    
    # Test forward pass
    output = model(sample_input)
    assert output.shape == sample_input.shape
    
    # Test state initialization
    model.init_states(sample_input)
    for layer in model.layers:
        assert hasattr(layer, 'ct_1')
        assert hasattr(layer, 'nt_1')

def test_XLSTMLanguageModel():
    vocab_size = 1000
    x = torch.randn(32, 128, 256)  # (batch_size, sequence_length, embedding_dim)
    model = XLSTMLanguageModel(vocab_size, x, layers=['s', 'm'])
    
    # Test initialization
    assert model.vocab_size == vocab_size
    assert model.n_embd == 256
    assert model.block_size == 128
    assert len(model.xlstm.layers) == 2
    
    # Test forward pass
    idx = torch.randint(0, vocab_size, (32, 64))  # (batch_size, sequence_length)
    logits, loss = model(idx)
    assert logits.shape == (32, 64, vocab_size)
    assert loss is None
    
    # Test with targets
    targets = torch.randint(0, vocab_size, (32, 64))
    logits, loss = model(idx, targets)
    assert logits.shape == (32, 64, vocab_size)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    
    # Test generation
    generated, last_token = model.generate(idx, max_new_tokens=10)
    assert generated.shape == (32, 74)  # original 64 + 10 new tokens
    assert last_token.shape == (32, 1)

def test_mLSTMblock(sample_input):
    model = mLSTMblock(sample_input, factor=2, depth=4)
    
    # Test initialization
    assert model.input_size == 256
    assert model.hidden_size == 512
    
    # Test forward pass
    output = model(sample_input)
    assert output.shape == sample_input.shape
    
    # Test state initialization
    model.init_states(sample_input)
    assert model.ct_1.shape == (1, 1, 512)
    assert model.nt_1.shape == (1, 1, 512)

def test_sLSTMblock(sample_input):
    model = sLSTMblock(sample_input, depth=4)
    
    # Test initialization
    assert model.input_size == 256
    
    # Test forward pass
    output = model(sample_input)
    assert output.shape == sample_input.shape
    
    # Test state initialization
    model.init_states(sample_input)
    assert model.ct_1.shape == (1, 1, 256)
    assert model.nt_1.shape == (1, 1, 256)
    assert model.ht_1.shape == (1, 1, 256)
    assert model.mt_1.shape == (1, 1, 256)

if __name__ == "__main__":
    pytest.main([__file__])