## A Seq2seq Model with Attention Mechanism in Neural Machine Translation

——by ccr

20.1.29

### Network Architecture

seq2seq model with global attention.

### Usage

Run `python3 run.py` to initialize a training. The training process saves the models to `NMT.ckpt` if an exception occurs or receives a `KeyBoardInterrupt`. It also automatically saves the models to `NMTtmp.pt` once several batches have done. Also the models are saved to `NMT.pt` after the whole training process has been done.

Run `python3 run.py <CheckPointFile>` to resume training from certain checkpoint.

Run `python3 evalModel.py <ModelFile>` to evaluate certain models.

### File Structure

`run.py` is the main file where training occurs.

`models.py` defines the encoder and the attention decoder model.

`evalModel.py` provides a way to evaluate the trained models.

`./en_vi` stores the raw data of English/Vietnamese dataset.

### Results

BLEU score on valid dataset: 25.27

BLEU score on test dataset: 22.65

See `training_procedure.txt` and `results.txt` for more details.