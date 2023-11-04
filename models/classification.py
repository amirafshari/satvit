import torch
import torch.nn as nn
import torch.nn.functional as F





class Classification(nn.Module):
    def __init__(self, img_height=9, img_width=9, in_channel=10,
                       patch_size=3, embed_dim=128, max_time=60,
                       num_classes=20, num_head=8, dim_feedforward=2048,
                       num_layers=8
                ):
        super().__init__()
        
        self.H = img_height
        self.W = img_width
        self.P = patch_size
        self.C = in_channel
        self.d = embed_dim
        self.T = max_time
        self.K = num_classes

        self.d_model = self.d
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers


        self.N = int(self.H * self.W // self.P**2)
        # self.n = int(self.N**0.5)
        self.nh = int(self.H / self.P)
        self.nw = int(self.W / self.P)

        self.encoderLayer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_head, dim_feedforward=self.dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=self.num_layers)

        self.projection = nn.Conv3d(self.C, self.d, kernel_size=(1, self.P, self.P), stride=(1, self.P, self.P))


        self.temporal_emb = nn.Linear(366, self.d)
        self.temporal_cls_token = nn.Parameter(torch.randn(1, self.K, self.d)) # (1, K, d)
        self.temporal_transformer = self.encoder


        self.spatial_emb = nn.Parameter(torch.randn(1, self.N, self.d)) # (1, N, d)
        self.spatial_cls_token = nn.Parameter(torch.randn(1, self.K, self.d)) # (1, K, d)
        self.spatial_transformer = self.encoder
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.d),nn.Linear(self.d, 1))


    def forward(self, x):

        '''
        Tekenization
        '''
        # remove the timestamps (last channel) from the input
        x_sits = x[:, :, :-1]
        B, T, C, H, W = x_sits.shape # (B, T, C, H, W)
        
        x_sits = x_sits.reshape(B, C, T, H, W) # (B, C, T, H, W)
        x_sits = self.projection(x_sits) # (B, d, T, nw, nh)
        x_sits = x_sits.view(B, self.d, T, self.nh*self.nw) # (B, d, T, N)

        # Spatial Encoding (Positional Embeddings)
        # we dont add pos embedding here, cuz we need the pure data for the temporal encoder
        # x_sits = x_sits + self.pos_emb # (B, d, T, N) 

        x_sits = x_sits.permute(0,3,2,1) # (B, N, T, d)



        '''
        Temporal Encoding
        '''
        # in the last channel lies the timestamp
        xt = x[:, :, -1, 0, 0] # (B, T, C, H, W)
        # convert to one-hot
        xt = F.one_hot(xt.to(torch.int64), num_classes=366).to(torch.float32) # (B, T, 366)
        Pt = self.temporal_emb(xt) # (B, T, d)


        '''
        Temporal Encoder: cat(Z+Pt)
        '''
        x = x_sits + Pt.unsqueeze(1) # (B, N, T, d)
        temporal_cls_token = self.temporal_cls_token # (1, K, d)
        temporal_cls_token = temporal_cls_token.repeat(B, self.N, 1, 1) # (B, N, K, d)
        x = torch.cat([temporal_cls_token, x], dim=2) # (B, N, K+T, d)
        x = x.view(B*self.N, self.K + T, self.d)
        x = self.temporal_transformer(x) # (B*N, K+T, d)
        x = x.view(B, self.N, self.K + T, self.d) # (B, N, K+T, d)
        x = x[:,:,:self.K] # (B, N, K, d)
        x = x.reshape(B, self.K, self.N, self.d) # (B, K, N, d)
        '''
        Spatial Encoding
        '''
        Ps = self.spatial_emb # (1, N, d)
        Ps = Ps.unsqueeze(1) # (1, 1, N, d)
        x = x + Ps # (B, K, N, d)

        spatial_cls_token = self.spatial_cls_token # (1, K, d)
        spatial_cls_token = spatial_cls_token.unsqueeze(2) # (1, K, 1, d)
        spatial_cls_token = spatial_cls_token.repeat(B, 1, 1, 1) # (B, 1, 1, d)

        x = torch.cat([spatial_cls_token, x], dim=2) # (B, K, N + 1, d)
    
        x = x.view(B*(self.N+1), self.K, self.d) # (B*(N+1), K, d)
        x = self.spatial_transformer(x) # (B*(N+1), K, d)
        x = x.view(B, self.N+1, self.K, self.d) # (B, (N+1), K, d)
        classes = x[:,0,:,:] # (B, K, d)
        x = self.mlp_head(classes) # (B, K, 1)
        x = x.reshape(B, -1) # (B, K)
                
        return x