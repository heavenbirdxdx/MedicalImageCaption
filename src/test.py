import torch
from torchvision.models import resnet50
import torch.nn as nn
import cv2 as cv
import numpy as np
from transformers import BertTokenizer
from model.Seq2Seq import Seq2Seq
from model.Decoder import Decoder
from model.ImageEncoder import ImageEncoder
from model.GraphEncoder import GCNEncoder
from utils import load_data
from config import Config
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def generate_caption(image_path, trg_tokenizer, model, device, max_len=100):
    model.eval()
    # print(sentence)
    
    img = cv.imread(image_path)
    image = cv.resize(img, (224, 224))
    image = np.float32(image) / 255.0
    image[:,:,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:,:,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((2, 0, 1))
    image_encoder = torch.from_numpy(image).to(device)

    image_encoder = image_encoder.unsqueeze(0)

    print(image_encoder.shape)

    # src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src1 = model.imageencoder(image_encoder)
        enc_src2 = model.graphencoder(features, adj)

        enc_src2 = enc_src2.permute(1,0)
        enc_src2 = model.fc(enc_src2)
        enc_src2 = enc_src2.permute(1,0)
        # print(enc_src1.shape, enc_src2.shape)
        # 融合
        # print(batch_size)
        enc_src = enc_src1+enc_src2[0:1,:]
    
    src_mask = model.make_src_mask(enc_src)

    enc_src = enc_src.unsqueeze(1)
    enc_src = enc_src.repeat(1,256,1)

    print(enc_src.shape)

    trg_indexes = [101]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == 102:
            break
    print(trg_indexes)
    trg_tokens = [trg_tokenizer.convert_ids_to_tokens(i) for i in trg_indexes]

    return trg_tokens[1:-1], attention


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")

    image_model = resnet50(pretrained=True)
    image_model.to(device)

    SRC_tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")
    TRG_tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")

    SRC_PAD_IDX = SRC_tokenizer.convert_tokens_to_ids(SRC_tokenizer.pad_token)
    TRG_PAD_IDX = TRG_tokenizer.convert_tokens_to_ids(TRG_tokenizer.pad_token)

    OUTPUT_DIM = len(TRG_tokenizer.vocab)

    opt = Config()

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    adj = adj.to(device)
    features = features.to(device)
    
    encoder = ImageEncoder(image_model)

    imageencoder = ImageEncoder(image_model)
    graphencoder = GCNEncoder(nfeat=features.shape[1],
            nhid=opt.hidden,
            nclass=opt.graph_output,
            dropout=opt.dropout)

    decoder = Decoder(OUTPUT_DIM,
              opt.HID_DIM,
              opt.DEC_LAYERS,
              opt.DEC_HEADS,
              opt.DEC_PF_DIM,
              opt.DEC_DROPOUT,
              device)

    model = Seq2Seq(imageencoder,graphencoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device)
    print(model.parameters)
    # model = Seq2Seq(tokenizer, opt, image_model, device)
    model.load_state_dict(torch.load("/home/xdx/project-MedicalImageCpation/checkpoints/250-0.1563-model.pt"))
    model.eval()
    model = model.to(device)
    # /home/xdx/MedicalImageCaption/data2/色素痣/202108240011 韩玥馨/皮肤镜/韩_玥馨_203998.JPG
    # /home/xdx/MedicalImageCaption/data2/基底细胞癌/202108050011 吴春珍/皮肤镜/吴_春珍_201615.JPG
    # /home/xdx/MedicalImageCaption/data2/黑色素瘤/202106100014 梁庆伍/皮肤镜/梁_庆伍_195196.JPG
    # /home/xdx/MedicalImageCaption/data2/黑色素瘤/202108120004 崔卫萍/皮肤镜/崔_卫萍_202464.JPG
    # /home/xdx/MedicalImageCaption/data2/基底细胞癌/202109240009 高岚/皮肤镜/高_岚_207559.JPG
    # /home/xdx/MedicalImageCaption/data2/黑色素瘤/202106210003 何亚卿/皮肤镜/何_亚卿_196261.JPG
    # /home/xdx/MedicalImageCaption/data2/色素痣/202108170002 顾银花/皮肤镜/顾_银花_203080.JPG
    image_path = "/home/xdx/MedicalImageCaption/data2/色素痣/202109100015 李晏菁/皮肤镜/李_晏菁_206213.JPG"

    seq, _ = generate_caption(image_path, TRG_tokenizer, model, device, max_len=100)

    res = "".join(seq)
    print(res)

    # res = ["".join([tokenizer.convert_ids_to_tokens(id) for id in complete_seq]) for complete_seq in complete_seqs]

    # print(image_path)
    # for r in res:
    #     print(r)