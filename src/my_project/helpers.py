import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q (torch.Tensor): Queries of shape (batch_size, num_heads, seq_len, d_k).
        K (torch.Tensor): Keys of shape (batch_size, num_heads, seq_len, d_k).
        V (torch.Tensor): Values of shape (batch_size, num_heads, seq_len, d_v).
        mask (torch.Tensor): Optional mask of shape (batch_size, 1, seq_len, seq_len).

    Returns:
        torch.Tensor: Output of attention mechanism.
        torch.Tensor: Attention weights.
    """
    d_k = Q.size(-1)
    # Compute the attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Apply the mask
    
    # Normalize the scores to probabilities
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute the weighted sum
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

