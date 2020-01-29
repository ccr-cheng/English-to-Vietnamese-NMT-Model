import torch
import sys
import re

from run import NMTmodel, device
from run import valid_data, valid_iter, test_data, test_iter
from run import valid, evaluateRandomly, evaluate, cal_bleu_score

if len(sys.argv) < 2:
    path = input('Please enter model file path: ')
else:
    path = sys.argv[1]
modeldata = torch.load(path, map_location=device)
print(f'Loading models from {path} ...')
NMTmodel.load_state_dict(modeldata['model_state'])
NMTmodel.eval()
print('Loading models complete!\n')

print('Testing on valid set ...')
valid_loss = valid(valid_iter)
print('Calulating BLEU score on valid set ...')
valid_bleu = cal_bleu_score(valid_data)
print('Testing on test set ...')
test_loss = valid(test_iter)
print('Calulating BLEU score on test set ...')
test_bleu = cal_bleu_score(test_data)

print(f'Valid:\tLoss: {valid_loss:.3f}\tBLEU: {valid_bleu:.2f}')
print(f'Test:\tLoss: {test_loss:.3f}\tBLEU: {test_bleu:.2f}')
print('Evaluating on test set...\n')
evaluateRandomly(test_data)


def cleanString(s):
    s = s.lower().strip()
    s = re.sub(r'([,.:;!?])', r' \1', s)
    s = re.sub(r'\'', r' &apos;', s)
    s = re.sub(r'\"', r' &quot; ', s)
    return s.split(' ')


print('Please enter sentence to be translated:\n')
while True:
    output_sent = evaluate(cleanString(input()))
    print('<', ' '.join(output_sent), '\n')
