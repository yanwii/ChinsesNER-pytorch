# ChinsesNER-pytorch

### train

setp 1: edit **models/config.yml**

    embedding_size: 100
    hidden_size: 128
    model_path: models/
    batch_size: 20
    dropout: 0.5
    tags:
      - ORG
      - PER

step 2: train

    python3 main.py train
    or
    cn = ChineseNER("train")
    cn.train()

    ...
    epoch [4] |██████                   | 154/591
            loss 0.46
            evaluation
            ORG     recall 1.00     precision 1.00  f1 1.00
    --------------------------------------------------
    epoch [4] |██████                   | 155/591
            loss 1.47
            evaluation
            ORG     recall 0.92     precision 0.92  f1 0.92
    --------------------------------------------------
    epoch [4] |██████                   | 156/591
            loss 0.46
            evaluation
            ORG     recall 0.94     precision 1.00  f1 0.97

### predict

    python3 main.py predict
    or 
    cn = ChineseNER("predict")
    cn.predict()

    请输入文本: 海利装饰材料有限公司
    [{'start': 0, 'stop': 10, 'word': '海利装饰材料有限公司', 'type': 'ORG'}]

### REFERENCES
- [Log-Linear Models, MEMMs, and CRFs](http://www.cs.columbia.edu/~mcollins/crf.pdf)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
