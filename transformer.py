import torch
from torch import nn
from torch import Tensor
import numpy as np

import torch.nn.functional as F


class Embedding(nn.Module):
    """
    Переводит токены в эмбеддинги Трансформера,
    суммируя эмбеддинги токенов и их позиций
    """

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob=0.1):
        """
        vocab_size: размер словаря
        hidden_size: размер скрытого слоя
        max_len: максимальная возможная длина текста
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)

        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, input_ids) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        """
        assert input_ids.ndim == 2
        tok_emb = self.word_embeddings(input_ids)

        position_ids = self.position_ids[:, :input_ids.shape[1]]
        pos_emb = self.position_embeddings(position_ids)

        emb = tok_emb + pos_emb
        return self.dropout(self.layer_norm(emb))


class MultiHeadAttention(nn.Module):
    """
    Реализует Multi-Head Self-Attention слой Трансформера.
    """
    def __init__(self, hidden_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        assert hidden_size % n_head == 0

        self.hidden_size = hidden_size
        self.n_head = n_head
        self.d_head = hidden_size // n_head

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        #self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_prob)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, attention_mask=None) -> Tensor:
        """
        q, k, v: Tensor[bs, seq_len, hidden_size] – входные аргументы для соответствующих линейных слоев
        attention_mask: Tensor[bs, 1, 1 or seq_len, seq_len] – маска внимания, содержащая значения 0 и -inf;
                                                              добавляется к скорам механизма внимания (до softmax)
        """
        assert q.ndim == k.ndim == v.ndim == 3
        assert attention_mask.ndim == 4

        bs = q.size(0)

        # 1. Cчитаем матрицы K, Q и V и разделяем их на головы.
        # Размерность каждой матрицы: [bs, seq_len, n_head, head_dim]
        # но также надо учесть transpose для n_head и seq_len, чтобы каждая голова была независима
        # по итогу [bs, n_head, seq_len, head_dim]
        K = self.key(k)
        K = K.view(K.shape[0], K.shape[1], self.n_head, -1).transpose(1, 2)

        Q = self.query(q)
        Q = Q.view(Q.shape[0], Q.shape[1], self.n_head, -1).transpose(1, 2)

        V = self.value(v)
        V = V.view(V.shape[0], V.shape[1], self.n_head, -1).transpose(1, 2)
        '''
        K = self.key(k[:, :, None, :].repeat(1,1,self.n_head,1))
        Q = self.query(q[:, :, None, :].repeat(1,1,self.n_head,1))
        V = self.value(v[:, :, None, :].repeat(1,1,self.n_head,1))
        '''
        # 2. Считаем attention_scores: Q * K^T / sqrt{head_dim}
        # Размерность результата: [bs, n_head, seq_len, seq_len]
        attn_sc = torch.matmul(Q, K.transpose(2, 3)) / (self.d_head ** 0.5)
        # 3. Добавляем attention_mask к полученным скорам, чтобы занулить те из них, на которые нельзя смотреть
        if attention_mask is not None:
          attn_sc = attn_sc + attention_mask
        # 4. Считаем attention_probs: softmax(attention_scores)
        # Softmax применяем к последней размерности
        attention_probs = F.softmax(attn_sc, dim=-1)
        # 5. Добавляем dropout к полученным вероятностям
        attention_probs = self.dropout(attention_probs)
        # 6. Считаем выход внимания: attention_probs * V
        # Размерность результата: [bs, n_head, seq_len, head_dim]
        attention_output = torch.matmul(attention_probs, V)
        # 7. Конкатенируем обратно векторы всех голов, получаем размерность [bs, seq_len, hidden_size]
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.hidden_size)
        # 8. Применяем последний линейный слой
        # Размерность результата: [bs, seq_len, hidden_size]
        return self.out(attention_output)

class FeedForward(nn.Module):
    """
    Реализует Feed Forward Network слой Трансформера c skip-connection и нормализацией.
    """
    def __init__(self, hidden_size, intermediate_size, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        # ваш код здесь
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.skip = nn.Identity()


    def forward(self, hidden_states) -> Tensor:
        """
        hidden_states: Tensor[bs, seq_len, hidden_size] – входное представление текста
        """
        assert hidden_states.ndim == 3

        # ваш код здесь
        x = self.relu(self.fc1(hidden_states))
        x = self.dropout(self.fc2(x))
        out = x + self.skip(hidden_states)
        return self.layer_norm(out)

class EncoderBlock(nn.Module):
    """
    Реализует блок Encoder'a.
    """
    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        # ваш код здесь
        self.att = MultiHeadAttention(hidden_size, n_head, drop_prob)
        self.ff = FeedForward(hidden_size, intermediate_size, drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.skip = nn.Identity()


    def forward(self, hidden_states, attention_mask) -> Tensor:
        """
        hidden_states: Tensor[bs, seq_len, hidden_size] – входное представление текста
        attention_mask: Tensor[bs, 1, 1, seq_len] – маска внимания, содержащая значения 0 и -inf
        """
        assert hidden_states.ndim == 3
        assert attention_mask.ndim == 4
        # ваш код здесь
        x = self.att(hidden_states, hidden_states, hidden_states, attention_mask)
        x = self.dropout(x)
        x = x + self.skip(hidden_states)
        x = self.layer_norm(x)
        return self.ff(x)


class Encoder(nn.Module):
    """
    Encoder Трансформера.
    """
    def __init__(self, vocab_size, max_len, hidden_size,
                 intermediate_size, n_head, n_layers, drop_prob=0.1):
        """
        vocab_size: размер словаря
        max_len: максимальная возможная длина текста
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        n_layers: число блоков Encoder
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()
        self.n_layers = n_layers
        # ваш код здесь
        self.emb = Embedding(vocab_size, hidden_size, max_len, drop_prob)
        self.enc_bl = nn.ModuleList([
            EncoderBlock(hidden_size, intermediate_size, n_head, drop_prob) for _ in range(n_layers)
            ])

    def forward(self, input_ids, attention_mask=None) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        attention_mask: Tensor[bs, 1, 1, seq_len] – маска внимания, содержащая значения 0 и -inf
        """
        assert input_ids.ndim == 2
        assert attention_mask.ndim == 4

        # ваш код здесь
        x = self.emb(input_ids)
        for enc in self.enc_bl:
          x = enc(x, attention_mask)
        return x

class DecoderBlock(nn.Module):
    """
    Реализует блок Decoder'a.
    """
    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        # ваш код здесь
        self.att1 = MultiHeadAttention(hidden_size, n_head, drop_prob)
        self.att2 = MultiHeadAttention(hidden_size, n_head, drop_prob)
        self.ff = FeedForward(hidden_size, intermediate_size, drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.skip = nn.Identity()

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        """
        hidden_states: Tensor[bs, trg_seq_len, hidden_size] – входное представление целевого текста
        attention_mask: Tensor[bs, 1, trg_seq_len, trg_seq_len] – маска внимания Decoder'a
        encoder_hidden_states: Tensor[bs, src_seq_len, hidden_size] – выход последнего слоя Encoder
        encoder_attention_mask: Tensor[bs, 1, 1, src_seq_len] – маска внимания Encoder'a
        """
        assert hidden_states.ndim == encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4

        # ваш код здесь
        x = self.att1(hidden_states, hidden_states, hidden_states, attention_mask)
        x = self.dropout(x)
        x = x + self.skip(hidden_states)
        x = self.layer_norm1(x)

        x2 = self.att2(x, encoder_hidden_states, encoder_hidden_states, encoder_attention_mask)
        x2 = self.dropout(x2)
        x2 = x2 + self.skip(x)
        x2 = self.layer_norm2(x2)

        return self.ff(x2)



class Decoder(nn.Module):
    """
    Decoder Трансформера.
    """
    def __init__(self, vocab_size, max_len, hidden_size,
                 intermediate_size, n_head, n_layers, drop_prob=0.1):
        """
        vocab_size: размер словаря
        max_len: максимальная возможная длина текста
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        n_layers: число блоков Decoder
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        # ваш код здесь
        self.n_layers = n_layers
        # ваш код здесь
        self.emb = Embedding(vocab_size, hidden_size, max_len, drop_prob)
        self.dec_bl = nn.ModuleList([
            DecoderBlock(hidden_size, intermediate_size, n_head, drop_prob) for _ in range(n_layers)
            ])
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        attention_mask: Tensor[bs, 1, trg_seq_len, trg_seq_len] – маска внимания Decoder'a
        encoder_hidden_states: Tensor[bs, src_seq_len, hidden_size] – выход последнего слоя Encoder
        encoder_attention_mask: Tensor[bs, 1, 1, src_seq_len] – маска внимания Encoder'a
        """
        assert input_ids.ndim == 2
        assert encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4

        # ваш код здесь
        # ваш код здесь
        x = self.emb(input_ids)
        for dec in self.dec_bl:
          x = dec(x, attention_mask, encoder_hidden_states, encoder_attention_mask)
        return self.out(x)

def get_extended_attention_mask(attention_mask, dtype=torch.float):
    # ваш код здесь
    attention_mask = attention_mask.to(torch.float)
    attention_mask[attention_mask==0] = torch.finfo(torch.float).min
    attention_mask[attention_mask==1] = 0
    extended_attention_mask = attention_mask[:,None,None,:]#.shape
    return extended_attention_mask


def get_causal_extended_attention_mask(attention_mask, dtype=torch.float):
    # ваш код здесь
    # сначала растянем исходную матрицу, где паддинг был - значение 2
    a1 = attention_mask.repeat(1, attention_mask.shape[1]).view(attention_mask.shape[0], attention_mask.shape[1], -1)
    a1[a1==0] = 2
    a1[a1==1] = 0
    # теперь создадим верхнедиагональную матрицу с единицами
    a2 = torch.ones_like(a1)
    a2 = torch.triu(a2, diagonal=1)

    # соединим паддинги и диагональную
    extended_attention_mask = a2 + a1

    # заменим положительные значения на -inf
    extended_attention_mask = extended_attention_mask.to(dtype)
    extended_attention_mask[extended_attention_mask > 0] = torch.finfo(dtype).min

    return extended_attention_mask[:, None, :, :]

class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, hidden_size, n_head,
                 intermediate_size, encoder_max_len, decoder_max_len, n_layers, drop_prob=0.1):
        super().__init__()
        """
        Все параметры означают то же самое, что и в предыдущих классах
        """
        # ваш код здесь
        self.encoder = Encoder(encoder_vocab_size, encoder_max_len, hidden_size, intermediate_size, n_head, n_layers, drop_prob)
        self.decoder = Decoder(decoder_vocab_size, decoder_max_len, hidden_size, intermediate_size, n_head, n_layers, drop_prob)


    def forward(self, src_input_ids, trg_input_ids, src_attention_mask=None, trg_attention_mask=None) -> Tensor:
        """
        src_input_ids: Tensor[bs, src_seq_len] – индексы токенов входного текста
        trg_input_ids: Tensor[bs, trg_seq_len] – индексы токенов выходного текста
        src_attention_mask: Tensor[bs, scr_seq_len] – маска внимания входного текста
        trg_attention_mask: Tensor[bs, trg_seq_len] – маска внимания выходного текста
        """

        # ваш код здесь
        if src_attention_mask is None:
            src_attention_mask = torch.ones(src_input_ids.shape, device=src_input_ids.device)
        src_attention_mask = get_extended_attention_mask(src_attention_mask)

        if trg_attention_mask is None:
            trg_attention_mask = torch.ones(trg_input_ids.shape, device=trg_input_ids.device)
        trg_attention_mask = get_causal_extended_attention_mask(trg_attention_mask)

        x = self.encoder(src_input_ids, src_attention_mask)
        x = self.decoder(trg_input_ids, trg_attention_mask, x, src_attention_mask)
        return x

    @torch.inference_mode()
    def generate(self, src_input_ids, src_attention_mask, trg_attention_mask, cls_token_id, sep_token_id, pad_token_id, max_length=40):
        """
        Жадно генерирует текст.
        """
        #print(src_attention_mask.shape)
        #src_attention_mask = src_attention_mask[:,None,None,:]
        #trg_attention_mask = trg_attention_mask[:,None,None,:]


        bs = len(src_input_ids)

        output_ids = torch.full((bs, 1), cls_token_id, device=src_input_ids.device)
        # инициализируем маску внимания для декодера
        trg_attention_mask = torch.ones(output_ids.shape, device=src_input_ids.device)

        #trg_attention_mask = get_causal_extended_attention_mask(trg_attention_mask).to(device)

        i = 0
        finished = torch.zeros(bs, device=src_input_ids.device)
        while i < max_length:
            logits = self.forward(src_input_ids, output_ids,
                       src_attention_mask, trg_attention_mask)
            next_tokens = logits[:, -1].argmax(-1)
            finished += next_tokens == sep_token_id
            if torch.all(finished > 0):
                break

            i += 1
            output_ids = torch.cat((output_ids, next_tokens.unsqueeze(-1)), dim=1)
            # обновляем маску декодера - добавляем паддинги
            trg_attention_mask = torch.ones(output_ids.shape, device=src_input_ids.device)
            trg_attention_mask[output_ids == pad_token_id] = 0
            #trg_attention_mask.to(device)
            #trg_attention_mask = get_causal_extended_attention_mask(trg_attention_mask_temp).to(device)
        return output_ids
