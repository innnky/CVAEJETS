import re
from text.japanese import japanese_to_phoneme
from text.mandarin import chinese_to_phoneme


def zh_ja_mixture_cleaners(text):
    chinese_texts = re.findall(r'\[ZH\].*?\[ZH\]', text)
    japanese_texts = re.findall(r'\[JA\].*?\[JA\]', text)
    phonemes = []
    temp = []
    for chinese_text in chinese_texts:
        temp.append((text.find(chinese_text), chinese_to_phoneme(chinese_text[4:-4])))
    for japanese_text in japanese_texts:
        temp.append((text.find(japanese_text), japanese_to_phoneme(japanese_text[4:-4]) ) )
    for _, ph in sorted(temp):
        phonemes += ph
    phonemes = [i for i in phonemes if i != '']
    return phonemes

if __name__ == '__main__':
    # print(zh_ja_mixture_cleaners("[ZH]這 太 可愛 了[ZH][JA]えー、字幕ゴミさんが[JA][ZH]也只到8月30号之前，就[ZH][JA]ヒーローイ![JA]"))
    symbols = set()
    with open("/Users/xingyijin/Documents/github-projects/CVAEJETS/dataset/raw_out_all.txt") as f:
        for line in f.readlines():
            _, _, text = line.strip().split("|")
            phs = zh_ja_mixture_cleaners(text)
            text_ = [symbols.add(ph) for ph in phs]
            if "。" in phs:
                print(phs)
        # [print(i) for i in symbols]
        print(sorted(symbols))