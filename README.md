## A Seq2seq Model with Attention Mechanism in Neural Machine Translation

——by ccr

20.1.31

### Network Architecture

Transformer model, with positional encoding.

### Usage

Run following to start training.

```bash
python3 main.py train
```

The script accepts following optional argument:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --lr                  initial learning rate [default: 5e-4]
  --lr-decay            learning rate decay factor [default: 0.5]
  --patience            patience before lr decay [default: 3]
  --num-epoch           number of training epochs [default:20]
  --num-layer           number of layers in encoder and decoder [default: 6]
  --batch-size          batch size [default: 64]
  --embed-size          embedding size (d_model) [default: 256]
  --hidden-size         hidden size (in the feedforward layers) [default: 512]
  --max-length          max length to trim the dataset [default: 64]
  --clip-grad           parameter clipping threshold [default: 1.0]
  --print-every         print training procedure every number of batches [default: 100]
  --save-path           model path for saving  [default: 'NMT.pt']
  --checkpoint          checkpoint for resuming training [default: '']
```

Run following to start testing.

```bash
python3 main.py test --model [model file]
```

### File Structure

`main.py` is the main file where training and testing occurs.

`transformerModel.py` defines the Transformer model.

`./en_vi` stores the raw data of English/Vietnamese dataset.

### Results

BLEU score on valid dataset: 26.81

BLEU score on test dataset: 24.77
