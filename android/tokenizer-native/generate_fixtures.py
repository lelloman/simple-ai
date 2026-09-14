"""Generate cross-binding fixtures with Python tokenizers==0.22.2."""
import json
from pathlib import Path
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

out = Path(__file__).resolve().parents[1] / 'app/src/test/resources/tokenizers'
out.mkdir(parents=True, exist_ok=True)
texts = ['Hello John', 'Caffè e città', 'مرحبا بالعالم', '你好世界', 'Hello 😀 John', 'Cafe\u0301', '  Hello, John!  ']
special = ['<s>', '<pad>', '</s>', '<unk>']
alphabet = sorted(set(''.join(texts) + '▁'))
vocab = [(s, 0.0) for s in special] + [(c, -10.0) for c in alphabet]
vocab += [('▁Hello', -1.0), ('▁John', -1.0), ('▁città', -1.0), ('▁你好世界', -1.0)]
unigram = Tokenizer(models.Unigram(vocab, unk_id=3))
unigram.normalizer = normalizers.NFC()
unigram.pre_tokenizer = pre_tokenizers.Metaspace()
bpe = Tokenizer(models.BPE(unk_token='<unk>'))
bpe.normalizer = normalizers.NFC()
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=300, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
for name, tokenizer in [('unigram', unigram), ('bpe', bpe)]:
    tokenizer.post_processor = processors.TemplateProcessing(single='<s> $A </s>', special_tokens=[('<s>', 0), ('</s>', 2)])
    tokenizer.enable_truncation(max_length=64)
    tokenizer.enable_padding(length=64, pad_id=1, pad_token='<pad>')
    tokenizer.save(str(out / f'{name}.json'))
    cases = []
    for text in texts:
        e = tokenizer.encode(text)
        def utf16(n): return len(text[:n].encode('utf-16-le')) // 2
        cases.append(dict(text=text, ids=e.ids, mask=e.attention_mask, offsets=[[utf16(a), utf16(b)] for a,b in e.offsets]))
    (out / f'{name}-expected.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
