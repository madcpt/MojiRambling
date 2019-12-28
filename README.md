# MojiRambling

## Configuration
**IMPORTANT:** make sure that you are using python>=3.6

`pip install reqiurements.txt`

## Pretrained Model
You can download the previous three models pretrained with **EMO_hash** dataset from https://drive.google.com/file/d/1xrTmI6pHEvqrvJWq6E3HLpd6bx_EbZ1I/view?usp=sharing.

After unzipping, you will get a ./save/ dir. place save/ in the main directory, with file structure like:

MojiRambling/

|__ ./save/

|__ ./data/

## DeepMoji (baseline)
pretrain: 
`python3 BaseLineRunner.py -mode=pretrain -n=1791 -cuda=1 -ds=data1_170000 -model=deepmoji -train=1`

fine-tune: 
`python3 BaseLineRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=deepmoji -train=1`

testing:
`python3 BaseLineRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=deepmoji -train=0`

## Transformer-Moji (proposed)
pretrain: 
`python3 TransformerMojiRunner.py -mode=pretrain -n=1791 -cuda=1 -ds=data1_170000 -model=transformermoji -train=1`

fine-tune: 
`python3 TransformerMojiRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=transformermoji -train=1`

testing:
`python3 TransformerMojiRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=transformermoji -train=0`

## Glove-Moji (proposed)
**Note:** First time you run this code, it may take a very long time, because the program will download Glove Embedding 
source file from a server in Stanford.

pretrain: 
`python3 GloveMojiRunner.py -mode=pretrain -n=1791 -cuda=1 -ds=data1_170000 -model=glovemoji -train=1`

fine-tune: 
`python3 GloveMojiRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=glovemoji -train=1`

testing:
`python3 GloveMojiRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=glovemoji -train=0`

## Bert-Moji (proposed)
**Note:** First time you run this code, it may take a very long time, because the program will download BERT pretrained 
model from Google Drive.

fine-tune: 
`python3 BertRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=bert -train=1`

## BertDeep-Moji (proposed)
**Note:** First time you run this code, it may take a very long time, because the program will download BERT pretrained 
model from Google Drive.

fine-tune: 
`python3 BertDeepRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=bertdeep -train=1`
