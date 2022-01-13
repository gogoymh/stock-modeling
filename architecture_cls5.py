import torch
from torch import nn
import math


# 총 3062 종목
# 하루 시가대비 종가 3프로 이상 상승 종목 개수: 최소 13종목, 최대 1951 종목, 평균 180 종목

# input : (B, 3062, 5) Batch size B개로 이루어진 하루동안 3062 종목의 5개 변수 시, 고, 저, 종, 거래량
# output : (B, 3062) Batch size B개로 이루어진 3062 종목 중 매매해야하는 종목 1, 하지 않아야하는 종목 0으로 나오는 벡터

def fourier_encoding(shape, bands):
    dims = len(shape)

    pos = torch.stack(list(torch.meshgrid(*(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape)))))
    org_pos = pos
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

    band_frequencies = (torch.logspace(
        math.log(1.0),
        math.log(shape[0]/2),
        steps=bands,
        base=math.e
    )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)

    result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

    result = torch.cat((
        torch.sin(result),
        torch.cos(result),
        org_pos
    ), dim=0)

    return result

       
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
      
class PreNorm(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x

class FNet(nn.Module):
    def __init__(self, d_model, depth, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, FNetBlock()),
                PreNorm(d_model, FeedForward(d_model, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class HB_transformer(nn.Module):
    def __init__(self, shape, device, output=1, d_models=32, depth=18, mlp_dim=16, dropout=0., fourier_band=4):
        super(HB_transformer, self).__init__()
        
        ## Hyperparamters ------------------------------------------------------------------------- ##
        if len(shape) == 3:
            channel, width, height = shape
            input_len = width * height
        elif len(shape) == 2:
            channel, input_len = shape
        else:
            NameError('Wrong shape')
        
        
        ## Positional Encoding ------------------------------------------------------------------------- ##
        self.encoding = fourier_encoding((input_len,), fourier_band).to(device)
        pec = self.encoding.shape[0]
        self.embed1 = nn.Linear(channel+pec, d_models)
                
        ## Layers ------------------------------------------------------------------------- ##
        self.model = FNet(d_models, depth, mlp_dim)
        
        self.norm = nn.LayerNorm([d_models])
        self.mlp = nn.Linear(d_models, output)
        self.gelu = nn.GELU()
        
    def forward(self, x): # (B, 3062, 5)
        B = x.shape[0]
        
        pos = self.encoding.unsqueeze(0).expand((B,) + self.encoding.shape) # (B, pec, 3062)
        pos = pos.permute(0,2,1) # (B, 3062, pec)
        
        x = torch.cat((x, pos), dim=-1) # (B, 3062, 5+pec)
        x = self.embed1(x) # (B, 3062, d_models)
        
        x = self.model(x)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.mlp(x)
        
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((5,1024,3)).to(device)
    b = HB_transformer((3,1024), device, output=10).to(device)
    
    c = b(a)
    print(c.shape)

    parameter = list(b.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)




















