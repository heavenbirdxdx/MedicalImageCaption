
from torch import dropout


class Config(object):

    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1


    encoder_dim = 2048
    decoder_dim = 128
    embedding_dim = 128
    attention_dim = 64
    rnn_hidden = 128
    num_layers = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCH = 300


    hidden = 512
    graph_output = 256
    dropout = 0.5