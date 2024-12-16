import helpers
import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Define learnable weight matrices
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
        self.dropout = torch.nn.Dropout(0.1)  # Optional dropout for regularization

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections for all heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention for each head
        attention_outputs, _ = helpers.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate and project back to d_model
        attention_outputs = attention_outputs.transpose(1, 2).contiguous()
        attention_outputs = attention_outputs.view(batch_size, -1, self.d_model)
        output = self.W_o(attention_outputs)
        
        return output

