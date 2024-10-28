# SNAC-Vocos
A trainer for [SNAC](https://github.com/hubertsiuzdak/snac) (Multi-Scale Neural Audio Codec) has replaced the decoder with Vocos.

## Installation
Suggested python>=3.9.  
Clone the repository:
```
git clone https://github.com/hertz-pj/SNAC-Vocos
cd SNAC-Vocos
```
Install packages:
```
pip install -r requirements.txt
```
## Infer
Refer to the [infer.py](./infer.py) for inference instructions and usage examples.

## Available Models
| Model name | Huggingface | Corpus | Domain |
|:------------|:--------|:--------|:--------|
|snac_vocos_16khz_hop200_scale8421_1kh | [🤗](https://huggingface.co/hertz-pj/snac-vocos) | 1k hours | Speech(Mandarin/English) |


## Training
1、Prepare a filelist of audio files for the training and validation set, e.g. [train.list](./data/train.list).  
2、Fill a config file, e.g. [snac_vocos.yaml](./config/snac_vocos_nq4_scale8421_16khz.yaml). The main parameters to pay attention to are batch_size, filelist_path, save_dir, and device.  
3、Start training
```
python train.py fit --config ./configs/snac_vocos.yaml
```

## TODO
- [x] Release code
- [x] Release a checkpoint trained with 1k hours of speech(Mandarin/English).
- [ ] Demo page.


## Acknowledgements
This implementation uses parts of the code from the following Github repos:  
- [SNAC](https://github.com/hubertsiuzdak/snac)
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer/)

