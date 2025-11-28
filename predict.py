import sys
import torch
import math

# YOUR CHANGES HERE
device = torch.device("cpu")

characters = "()+0123456789="

TOKENS = ["<bos>", "<eos>", "<pad>"] + [c for c in characters]
TOKEN_MAP = dict((t, i) for i, t in enumerate(TOKENS))

BOS = TOKEN_MAP["<bos>"]
EOS = TOKEN_MAP["<eos>"]
PAD = TOKEN_MAP["<pad>"]

def decode(token_ids):
    return "".join(TOKENS[i] for i in token_ids)

def encode(s, *, eos=True):
    if s.startswith("<bos>"):
        s = s[5:]

    output = [BOS]
    output.extend(TOKEN_MAP[c] for c in s)

    if eos:
        output.append(EOS)

    return torch.tensor(output, device=device)

def causal_mask(T):
    # shape (T, T); True = mask (disallow), False = keep
    # nn.Transformer expects float mask or bool depending on API;
    # TransformerEncoder uses src_mask where non-zero entries are masked.
    # We'll use a float mask with -inf above diagonal.
    m = torch.full((T, T), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)  # upper triangle is masked
    return m

class MathTransformer(torch.nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_ff=512, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        vocab_size = len(TOKENS)

        # token + position embeddings
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = torch.nn.Embedding(max_len, d_model)

        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True,   
            )
            for _ in range(num_layers)
        ])
        
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

        # init
        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.lm_head.bias)

    def forward(self, x):
        # x: (N, T)
        N, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)  # (N, T, d_model)

        # key padding mask: True where we want to ignore (PAD)
        key_padding_mask = (x == PAD)  # (N, T) bool

        # causal mask for self-attention (float, -inf above diagonal)
        attn_mask = causal_mask(T).to(x.device) # (T, T)

        for layer in self.blocks:
            h = layer(
                h,
                src_mask=attn_mask,                         # causal
                src_key_padding_mask=key_padding_mask   # pad masking
            )
        logits = self.lm_head(h)  # (N, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, prefix_ids, max_new_tokens=400):
        self.eval()
        x = prefix_ids.clone().to(next(self.parameters()).device)  # (N, T0)

        max_total_len = self.max_len
        for _ in range(max_new_tokens):
            if x.size(1) >= max_total_len:
                break
            logits = self.forward(x)[:, -1, :]   # (N, V)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
            x = torch.cat([x, next_id], dim=1)
            if (next_id == EOS).all():
                break
        return x 


def load_model(model_path):
    model = MathTransformer()
    state_dict = torch.load(model_path, map_location="cpu")  
    model.load_state_dict(state_dict)
    return model

def predict_completion(model, prompt):
    model.eval()
    input_ids = encode(prompt, eos=False).unsqueeze(0)  # (1, T)
    output_ids = model.generate(input_ids)[0]

    filtered_ids = [i for i in output_ids.cpu().numpy() if i != BOS and i != EOS and i != PAD]
    completion_str = "".join(TOKENS[i] for i in filtered_ids)
    return completion_str

def main(): 
    input_file = sys.argv[1]
    model_path = 'math.pt'  

    model = load_model(model_path)
    model.to(device)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            prompt = line.strip()
            if not prompt:
                continue
            completion = predict_completion(model, prompt)
            print(completion)

if __name__ == "__main__":
    main()