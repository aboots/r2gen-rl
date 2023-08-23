import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        elif self.dataset_name == 'mimic_cxr':
            self.clean_report = self.clean_report_mimic_cxr
        else:
            self.clean_report = self.clean_report_ffa_ir
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + \
            ['#unk#']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        def report_cleaner(t): return t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        def sent_cleaner(t): return re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                           replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(
            report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        def report_cleaner(t): return t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        def sent_cleaner(t): return re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                           .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(
            report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def first_preprocess(self, report):
        if report.startswith('1. '):
            report = report[3:]
        report = report.replace(';1', ' 1')
        report = re.sub(r'(\d+)PD', r'\1 PD', report)
        report = re.sub(r'(\d+(\.\d+)?)PD', r'\1 PD', report)
        report = report.replace('\u3001', '.')
        report = report.replace('\n', ' ').replace('=', ' = ').replace(
            '<', ' < ').replace('>', ' > ').replace('~', ' ~ ').replace('*', ' * ')
        return report

    def preprocess_nums(self, x):
        x = re.sub(r'\d+/\d+', '#NUM#', x)
        x = re.sub(r'\d+\.\d+s', '#NUM# seconds', x)
        x = re.sub(r'\d+s', '#NUM# seconds', x)
        x = re.sub(r'\d+\.\d+', '#NUM#', x)
        x = re.sub(r'(?<!\d)\d+\.\d+(?!\.)', '#NUM#', x)
        x = re.sub(r'(?<!\S)\d+(?!\S)', '#NUM#', x)

        return x

    def preprocess_time(self, text):
        words = text.split()
        result = []
        for i, word in enumerate(words):
            match = re.match(r'(\d{1,2}):(\d{1,2})', word)
            if match:
                if i > 0 and 'ratio' in words[i-3:i]:
                    result.append('#NUM#')
                else:
                    hour, minute = map(int, match.groups())
                    if 0 <= hour < 24 and 0 <= minute < 60:
                        result.append('#TIME#')
                    else:
                        result.append(word)
            else:
                result.append(word)
        return ' '.join(result)

    def clean_report_ffa_ir(self, report):
        if report.startswith('1. '):
            report = report[3:]
        report = re.sub(r'(\d+(\.\d+)?)PD', r'\1 PD', report)
        report = report.replace('\u3001', '.')

        def report_cleaner(t): return t.replace('\n', ' ').replace('=', ' = ').replace('<', ' < ').replace('>', ' > ').replace('~', ' ~ ') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('\u3001', '.') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('1. ', ' ').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace('. 6. ', '. ') \
            .replace('. 7. ', '. ').replace('. 8. ', '. ').replace('. 9. ', '. ').replace('. 10. ', '. ') \
            .replace(' 1, ', '. ').replace(' 2, ', '. ').replace(' 3, ', '. ').replace(' 4, ', '. ').replace(' 5, ', '. ') \
            .replace(' 6, ', '. ').replace(' 7, ', '. ').replace(' 8, ', '. ').replace(' 9, ', '. ').replace(' 10, ', '. ') \
            .replace(' 1. ', '. ').replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .replace(' 6. ', '. ').replace(' 7. ', '. ').replace(' 8. ', '. ').replace(' 9. ', '. ').replace(' 10. ', '. ') \
            .replace('.1.', '').replace('.2.', '. ').replace('.3.', '. ').replace('.4.', '. ').replace('.5.', '. ') \
            .replace('.6.', '. ').replace('.7.', '. ').replace('.8.', '. ').replace('.9.', '. ').replace('.10.', '. ') \
            .replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '') \
            .replace('6.', '').replace('7.', '').replace('8.', '').replace('9.', '').replace('10.', '') \
            .replace('#num##num#', '#num#').replace('#num#-#num#', '#num#').replace('/', ' ') \
            .strip().lower().split('. ')

        def sent_cleaner(t): return t.translate(
            str.maketrans('', '', '!"$%&\'()+.,/:;?@[\\]^_`{|}'))
        tokens = [sent_cleaner(sent) for sent in report_cleaner(self.preprocess_time(
            self.preprocess_nums(self.first_preprocess(report)))) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        report = report.replace('.', '').replace('  ', ' ')
        return report.strip()

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['#unk#']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
