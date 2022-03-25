import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
# from model import Encoder, Decoder, Seq2Seq
from tqdm import tqdm
import math
from transformers import BertTokenizer
import time
# from torchtext.data import Field, BucketIterator
import json
import os
from PIL import Image
from torchvision.models import resnet50
import numpy as np
import cv2 as cv
import json
from model.Seq2Seq import Seq2Seq
from model.Decoder import Decoder
from model.ImageEncoder import ImageEncoder
from model.GraphEncoder import GCNEncoder
from utils import load_data
from config import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

def collate_func(batch_dic):

    batch_dic.sort(key=lambda d:len(d['caption_encoder']), reverse=True)
    batch_len=len(batch_dic) # 批尺寸
    max_length=max([len(dic['caption_encoder']) for dic in batch_dic])
    lengths = [min(len(dic["caption_encoder"])+1, max_length) for dic in batch_dic]
    
    src_batch=[]
    trg_batch=[]
    for i in range(len(batch_dic)): # 分别提取批样本中的feature、label、id、length信息
        dic=batch_dic[i]
        src_batch.append(dic['caption_encoder'])
        trg_batch.append(dic['image_encoder'])

    res={}
    res['caption_encoder']=pad_sequence(src_batch,batch_first=True) # 将信息封装在字典res中
    res['image_encoder']=pad_sequence(trg_batch,batch_first=True)
    res['lengths'] = lengths
    return res

class CaptionDataset(Dataset):
    def __init__(self, train_data_file_path,tokenizer_zh, device):
        super(CaptionDataset, self).__init__()
        with open(train_data_file_path, "r",encoding="utf-8") as fr:
            self.datas = json.load(fr)
        self.train_data_file_path = train_data_file_path
        self.tokenizer_zh = tokenizer_zh
        self.device = device

    def __getitem__(self, i):
        data = eval(str(self.datas[i]))
        image_url = data["image_url"]
        caption = data["caption"]
        caption_encoder = torch.tensor(self.tokenizer_zh.encode(caption)).to(self.device)
        img = cv.imread(image_url)
        image = cv.resize(img, (224, 224))
        image = np.float32(image) / 255.0
        image[:,:,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:,:,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        image = image.transpose((2, 0, 1))
        image_encoder = torch.from_numpy(image).to(device)
        sample = {"caption_encoder":caption_encoder, "image_encoder":image_encoder}
        return sample

    def __len__(self):
        return int(len(self.datas))

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_model = resnet50(pretrained=True)
    # modules = list(image_model.children())[:-2]
    # image_model = nn.Sequential(*modules)
    # # del image_model.fc
    # # image_model.fc = lambda x:x
    # image_model.to(device)

    SRC_tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")
    TRG_tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")

    SRC_PAD_IDX = SRC_tokenizer.convert_tokens_to_ids(SRC_tokenizer.pad_token)
    TRG_PAD_IDX = TRG_tokenizer.convert_tokens_to_ids(TRG_tokenizer.pad_token)

    OUTPUT_DIM = len(TRG_tokenizer.vocab)

    opt = Config()
    train_data_file_path = "/home/xdx/MedicalImageCaption/data2/pifujing_train.json"
    test_data_file_path = "/home/xdx/MedicalImageCaption/data2/pifujing_test.json"
    # image_root_path = "/home/xdx/ImageCaption/data/ai_challenger_caption_train_20170902/caption_train_images_20170902"

    # train_data_file_path = "/home/xdx/ImageCaption/data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"
    # image_root_path = "/home/xdx/ImageCaption/data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"

    # images_features,_ = torch.load('/home/xdx/ImageCaption/features/image_train_features.pth')
    # print(images_features.size())

    tokenizer_zh = BertTokenizer.from_pretrained("../bert-base-chinese")

    train_dataset = CaptionDataset(train_data_file_path,tokenizer_zh,device)
    train_dataloader = DataLoader(train_dataset, opt.BATCH_SIZE,collate_fn=collate_func, shuffle=True)

    test_dataset = CaptionDataset(test_data_file_path,tokenizer_zh,device)
    test_dataloader = DataLoader(test_dataset, int(opt.BATCH_SIZE/2),collate_fn=collate_func, shuffle=True)

    # 加载图数据
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

    model = Seq2Seq(imageencoder, graphencoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device)
    model = model.to(device)
    # model.load_state_dict(torch.load("/home/xdx/MedicalImageCaption/model/190-0.3278-model.pt"))
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-4)
    
    best_loss = 100

    for epoch in range(opt.EPOCH):
        epoch_train_loss = 0
        for batch_dic in tqdm(train_dataloader):
            caption_encoder = batch_dic["caption_encoder"]
            image_encoders = batch_dic["image_encoder"]
            lengths = batch_dic["lengths"]
            optimizer.zero_grad()


            batch_src_zh_id = image_encoders.to(device)
            batch_trg_en_id = caption_encoder.to(device)

            output, _ = model(features, adj, batch_src_zh_id, batch_trg_en_id[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = batch_trg_en_id[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            loss += loss.item()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        print("Epoch:{}\t Epoch_train_loss:{}".format(epoch, epoch_train_loss/len(train_dataloader)))

        epoch_test_loss = 0
        
        with torch.no_grad():
            for batch_dic in tqdm(test_dataloader):
                caption_encoder = batch_dic["caption_encoder"]
                image_encoders = batch_dic["image_encoder"]
                lengths = batch_dic["lengths"]
                # optimizer.zero_grad()

                batch_src_zh_id = image_encoders.to(device)
                batch_trg_en_id = caption_encoder.to(device)

                output, _ = model(features, adj, batch_src_zh_id, batch_trg_en_id[:, :-1])

                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = batch_trg_en_id[:, 1:].contiguous().view(-1)

                loss = criterion(output, trg)
                # loss.backward()
                # optimizer.step()
                epoch_test_loss += loss.item()
        print("Epoch:{}\t Epoch_test_loss:{}".format(epoch, epoch_test_loss/len(test_dataloader)))


        if epoch % 10 ==0 and epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            torch.save(model.state_dict(), '/home/xdx/project-MedicalImageCpation/checkpoints/{}-{:.4f}-model.pt'.format(epoch, best_loss/len(test_dataloader)))
            