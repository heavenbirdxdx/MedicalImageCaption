# from Decoder import Decoder
# from ImageEncoder import ImageEncoder
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, 
                 imageencoder,
                 graphencoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.imageencoder = imageencoder
        self.graphencoder = graphencoder
        self.fc = nn.Linear(117, 16)
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, features, adj, src, trg):
        
        batch_size = trg.shape[0]
        #src = [batch size, src len]
        #trg = [batch size, trg len]
        
        # print(src.shape)

        # src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src1 = self.imageencoder(src)
        enc_src2 = self.graphencoder(features, adj)

        enc_src2 = enc_src2.permute(1,0)
        enc_src2 = self.fc(enc_src2)
        enc_src2 = enc_src2.permute(1,0)
        # print(enc_src1.shape, enc_src2.shape)
        # 融合
        # print(batch_size)
        enc_src = enc_src1+enc_src2[0:batch_size,:]

        src_mask = self.make_src_mask(enc_src)

        # print(src_mask.shape)

        enc_src = enc_src.unsqueeze(1)

        enc_src = enc_src.repeat(1, 256, 1)
        
        # print(enc_src.shape)
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
