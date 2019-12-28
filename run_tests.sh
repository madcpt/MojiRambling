python3 BaseLineRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=deepmoji -train=0
python3 TransformerMojiRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=transformermoji -train=0
python3 GloveMojiRunner.py -mode=transfer -n=2 -cuda=1 -ds=SS-Youtube -model=glovemoji -train=0
