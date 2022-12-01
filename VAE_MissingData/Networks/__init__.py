from .conv_decoder import ConvDecoderMNIST
from .conv_encoder import ConvEncoderMNIST
from .conv_decoder_mask import ConvDecoderMaskMNIST

dic_network = {
    "ConvDecoderMNIST" : ConvDecoderMNIST,
    "ConvDecoderMaskMNIST": ConvDecoderMaskMNIST,
    "ConvEncoderMNIST" : ConvEncoderMNIST,
}