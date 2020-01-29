import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.embedding.weight.data)
        xavier_uniform_(self.fc_h.weight.data)
        xavier_uniform_(self.fc_c.weight.data)

    def forward(self, src, src_len):
        '''
        Forwarding
        :param src: source tensor with shape (max_src, b)
        :param src_len: lengths of src with shape (b)
        :return: outputs for attention with shape (b, max_src, 2h)
                 Last hidden state with shape (b, h)
                 Last cell state with shape (b, h)
        '''
        # embedded: (max_src, b, e)
        embedded = self.dropout(self.embedding(src))
        packed_embedded = pack_padded_sequence(embedded, src_len)

        # hidden: (2, b, h)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        # output: (b, max_src, 2h)
        outputs = pad_packed_sequence(packed_outputs)[0].transpose(0, 1)
        # hidden: (b, h)
        hidden = self.fc_h(torch.cat((hidden[0], hidden[1]), dim=1))
        # cell: (b, h)
        cell = self.fc_c(torch.cat((cell[0], cell[1]), dim=1))
        return outputs, (hidden, cell)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.embedding.weight.data)
        xavier_uniform_(self.attn.weight.data)
        xavier_uniform_(self.attn_combine.weight.data)
        xavier_uniform_(self.out.weight.data)

    def forward(self, trg, enc_state, enc_outputs, mask, inference=False):
        '''
        Forwarding
        :param trg: target tensor with shape (max_trg, b) or (1, b) for inference
        :param enc_state: initial state
        :param enc_outputs: tensor with shape (b, max_src, 2h)
        :param mask: tensor with shape (b, max_src)
        :param inference: whether doing inference
        :return: combine_output with shape (b, out_size, max_trg - 1)
        '''
        max_len = trg.size(0) if not inference else int(mask.size(1) * 1.5)

        # initial decoder state
        dec_state = enc_state
        # initial decode output dec_output: (b, h)
        dec_output = torch.zeros_like(dec_state[0], dtype=torch.float, device=trg.device)

        # attn can be computed once and used many times
        # attn: (b, max_src, h)
        attn = self.attn(enc_outputs)

        # embedded: (max_trg, b, e)
        embedded = self.dropout(self.embedding(trg))
        # dec_input: (b, e)
        dec_input = embedded[0]

        combine_output = []
        for idx in range(max_len - 1):
            # cat input and previous output to get new input
            # dec_input: (b, e + h)
            dec_input = torch.cat((dec_input, dec_output), 1)
            # new decoder state
            dec_state = self.lstm(dec_input, dec_state)
            # dec_hidden: (b, h)
            dec_hidden = dec_state[0]

            # attn_weights: (b, max_src)
            attn_weights = torch.bmm(attn, dec_hidden.unsqueeze(-1)).squeeze(-1)
            # set masked to -inf
            attn_weights = F.softmax(attn_weights.masked_fill(mask == 1, float('-inf')), 1)
            # attn_applied: (b, 2h)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

            # dec_output: (b, h)
            dec_output = self.attn_combine(torch.cat((attn_applied, dec_hidden), dim=1))
            # output: (b, out_size)
            output = self.out(F.relu(dec_output))

            combine_output.append(output)
            top1 = output.argmax(dim=1).detach()

            # next dec_input: (b, e)
            dec_input = self.embedding(top1) if inference else embedded[idx + 1]

        # combine_output: (b, out_size, max_trg - 1)
        combine_output = torch.stack(combine_output, -1)
        return combine_output


class seq2seq(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size,
                 pad_token, sos_token, eos_token, dropout_p=0.1):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.dropout_p = dropout_p

        self.encoder = EncoderRNN(input_size, embed_size, hidden_size, dropout_p)
        self.decoder = DecoderRNN(hidden_size, embed_size, output_size, dropout_p)

    @staticmethod
    def generate_mask(src, src_len):
        '''
        Generate mask for encoder hidden state
        :param src: source tensor with shape (max_src, b)
        :param src_len: Lengths of src with shape (b)
        :return: mask with shape (b, max_src) where pad_token is masked with 1
        '''
        masks = torch.zeros(src.size(1), src.size(0), dtype=torch.uint8).to(src.device)
        for idx, length in enumerate(src_len):
            masks[idx, src_len[idx]:] = 1
        return masks

    def forward(self, src, src_len, trg=None):
        '''
        Forwarding net on a batch of examples
        :param src: Source tensor with shape (max_src, b)
        :param src_len: Lengths of src with shape (b)
        :param trg: Target tensor with shape (max_trg, b)
        :return: Return output with shape (b, out_size, max_trg - 1)
        '''
        # enc_output: (max_src, b, 2h)
        enc_output, enc_state = self.encoder(src, src_len)
        # mask: (b, max_src)
        mask = self.generate_mask(src, src_len)

        inference = False
        # trg is None, doing inference
        # make trg filled with sos_token
        if trg is None:
            inference = True
            trg = torch.zeros(1, src.size(1), dtype=torch.long, device=src.device).fill_(self.sos_token)
        # initial hidden state is enc_state
        output = self.decoder(trg, enc_state, enc_output, mask, inference)
        return output


if __name__ == '__main__':
    EMBED_SIZE = 256
    INPUT_SIZE = 1024
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 1024

    NMTmodel = seq2seq(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, 0, 1, 2)
    print(NMTmodel)

    from torch.nn.utils.rnn import pad_sequence

    BATCH_SIZE = 16
    src_len, _ = torch.sort(torch.randint(1, 10, (BATCH_SIZE,)), descending=True)
    src = [torch.randint(1, 10, (length.item(),)) for length in src_len]
    src = pad_sequence(src, padding_value=0)
    trg_len = torch.randint(1, 10, (BATCH_SIZE,))
    trg = [torch.randint(1, 10, (length.item(),)) for length in trg_len]
    trg = pad_sequence(trg, padding_value=0)
    output = NMTmodel(src, src_len, trg)

    print('src size:', src.size())
    print('trg size:', trg.size())
    print('output size:', output.size())
