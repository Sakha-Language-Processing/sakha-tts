# Sakha Text to Speech

Преобразовываем якутскую речь в текст.

## Подготовка

Используйте Python 3.8 - 3.10.

```bash
wget "https://dl.fbaipublicfiles.com/mms/tts/sah.tar.gz"
tar -xzf sah.tar.gz

git clone https://github.com/jaywalnut310/vits
cp -r monotonic_align vits/monotonic_align/

python -m venv venv
. venv/bin/activate
pip install librosa==0.8 phonemizer torch
```

## Использование

```bash
python tts.py "Дорообо" "result.wav"
```

## Ссылки

* [The Massively Multilingual Speech (MMS) by Facebook Research](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
* [Готовые веса из проекта The Massively Multilingual Speech (MMS)](https://dl.fbaipublicfiles.com/mms/tts/sah.tar.gz)
* [VITS](https://github.com/jaywalnut310/vits)
