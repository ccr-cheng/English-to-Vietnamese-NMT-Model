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
  --lr                  initial learning rate
  --num-epoch           number of training epochs
  --batch-size          batch size
  --embed-size          embedding size (d_model)
  --hidden-size         hidden size (in the feedforward layers)
  --max-length          max length to trim the dataset
  --clip-grad           parameter clipping threshold
  --print-every         print training procedure every number of batches
  --save-every          save model every number of batches
  --save-path           model path for saving
  --checkpoint          checkpoint for resuming training
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

BLEU score on valid dataset: 26.45

BLEU score on test dataset: 24.06
