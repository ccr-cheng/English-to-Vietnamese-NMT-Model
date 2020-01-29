import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext.datasets import TranslationDataset

from tqdm import tqdm
import time, sys, random

from models import seq2seq

N_EPOCHS = 10
BATCH_SIZE = 32
MAX_LENGTH = 64
PRINT_EVERY = 100
SAVE_EVERY = 3000
LR = 0.001
CLIP_GRAD = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_len(example):
    return len(example.src) < MAX_LENGTH and len(example.trg) < MAX_LENGTH


SRC = Field(init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
TRG = Field(init_token='<sos>', eos_token='<eos>', lower=True)

print('Reading datasets ...')
test_data = TranslationDataset(path=r'./en_vi/test', exts=('.src', '.tgt'),
                               filter_pred=filter_len, fields=(SRC, TRG))
train_data = TranslationDataset(r'./en_vi/train', exts=('.src', '.tgt'),
                                filter_pred=filter_len, fields=(SRC, TRG))
valid_data = TranslationDataset(r'./en_vi/valid', exts=('.src', '.tgt'),
                                filter_pred=filter_len, fields=(SRC, TRG))
print('Datasets reading complete!')

print('Building vocabulary ...')
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print('Vocabulary building complete!')

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE, device=device,
    sort_within_batch=True, sort_key=lambda x: len(x.src)
)

EMBED_SIZE = 256
INPUT_SIZE = len(SRC.vocab)
OUTPUT_SIZE = len(TRG.vocab)
HIDDEN_SIZE = 256
SOS_token = TRG.vocab.stoi['<sos>']
EOS_token = TRG.vocab.stoi['<eos>']
PAD_token = TRG.vocab.stoi['<pad>']

NMTmodel = seq2seq(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                   PAD_token, SOS_token, EOS_token, dropout_p=0.1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token, reduction='sum').to(device)
optimizer = optim.Adam(NMTmodel.parameters(), lr=LR)
cur_batch = 0
cur_epoch = 0
start_time = time.time()


def train_epoch(print_every, save_every=1000, start_batch=0):
    global cur_batch
    epoch_loss = 0
    batch_num = len(train_iter)
    NMTmodel.train()

    with tqdm(total=print_every) as pbar:
        for i, batch in enumerate(train_iter):
            # skip trained
            if i < start_batch: continue
            cur_batch = i

            # src: (max_src, b)
            # src_len: (b)
            src, src_len = batch.src
            # trg: (max_trg, b)
            trg = batch.trg

            optimizer.zero_grad()
            # output: (b, out_size, max_trg - 1)
            output = NMTmodel(src, src_len, trg)

            # ignore the first sos_token
            loss = criterion(output, trg[1:].transpose(0, 1)) / (trg.size(0) - 1)
            epoch_loss += loss.item()
            loss.backward()

            clip_grad_norm_(NMTmodel.parameters(), CLIP_GRAD)
            optimizer.step()
            pbar.update(1)

            if (i + 1) % print_every == 0:
                secs = int(time.time() - start_time)
                mins = secs // 60
                secs = secs % 60

                print(f'\nEpoch {cur_epoch} |\t{i + 1}/{batch_num} batches done in '
                      f'{mins} minutes, {secs} seconds\t Batch loss: {loss:.4f}')
                evaluateRandomly(valid_data, 1)
                NMTmodel.train()
                pbar.reset()

            if (i + 1) % save_every == 0:
                print('Saving current model ...\n')
                saveModel('NMTtmp.pt')

    return epoch_loss / batch_num


def valid(data_iter):
    NMTmodel.eval()
    total_loss = 0
    batch_num = len(data_iter)
    with torch.no_grad():
        with tqdm(total=batch_num) as pbar:
            for i, batch in enumerate(data_iter):
                src, src_len = batch.src
                trg = batch.trg
                output = NMTmodel(src, src_len, trg)
                loss = criterion(output, trg[1:].transpose(0, 1)) / (trg.size(0) - 1)
                total_loss += loss.item()
                pbar.update(1)
    return total_loss / batch_num


def evaluate(sent):
    NMTmodel.eval()
    with torch.no_grad():
        src = [SOS_token] + [SRC.vocab.stoi[w] for w in sent] + [EOS_token]
        src = torch.tensor(src, dtype=torch.long, device=device).unsqueeze(-1)
        src_len = torch.tensor([src.size(0)], dtype=torch.long, device=device)

        output = NMTmodel(src, src_len)
        wordidx = torch.argmax(output.squeeze(0), dim=0)
        words = []
        for idx in wordidx:
            words.append(TRG.vocab.itos[idx])
            if idx == EOS_token: break

    return words


def evaluateRandomly(data: TranslationDataset, n=5):
    for i in range(n):
        example = random.choice(data.examples)
        src, trg = example.src, example.trg
        print('>', ' '.join(src))
        print('=', ' '.join(trg))
        output_sentence = evaluate(src)
        print('<', ' '.join(output_sentence))
        print()


def cal_bleu_score(data):
    trgs, preds = [], []
    num_example = len(data.examples)
    with tqdm(total=num_example) as pbar:
        for example in data.examples:
            src, trg = example.src, example.trg
            pred = evaluate(src)[:-1]
            trgs.append([trg])
            preds.append(pred)
            pbar.update(1)
    return bleu_score(preds, trgs) * 100


def saveModel(path='NMT.pt'):
    torch.save({
        'model_state': NMTmodel.state_dict(),
        'epoch': cur_epoch,
        'batch_num': cur_batch
    }, path)


def main():
    global cur_batch, cur_epoch

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f'Loading checkpoint from {path} ...')
        checkpoint = torch.load(path)
        NMTmodel.load_state_dict(checkpoint['model_state'])
        cur_epoch = checkpoint['epoch']
        cur_batch = checkpoint['batch_num']
        print(f'Checkpoint loading complete! Starting training from '
              f'epoch {cur_epoch}, batch {cur_batch}\n')

    try:
        for epoch in range(N_EPOCHS):
            # skip trained epochs
            if epoch < cur_epoch:
                continue
            cur_epoch = epoch

            start_epoch = time.time()
            train_loss = train_epoch(PRINT_EVERY, SAVE_EVERY, cur_batch)

            secs = int(time.time() - start_epoch)
            mins = secs // 60
            secs = secs % 60

            print('Checking performance on valid dataset...')
            valid_loss = valid(valid_iter)
            print(f'Epoch: {epoch}', f' | time in {mins} minutes, {secs} seconds')
            print(f'\tLoss: {train_loss:.4f}(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)')

            evaluateRandomly(valid_data)
            saveModel()
            cur_batch = 0

    except Exception as e:
        path = 'NMT.ckpt'
        print(f'A {type(e).__name__} occurs! Saving model to {path}')
        saveModel(path)
        raise

    except KeyboardInterrupt:
        path = 'NMT.ckpt'
        print(f'Training interrupted! Saving model to {path}')
        saveModel(path)
        exit(1)


if __name__ == '__main__':
    main()
