# Voice-Activated Video Tagging System

## 실행 

[](https://huggingface.co/spaces/famousdetectiveadrianmonk/whisper-small-nato-phonetic-alphabet)
에서 테스트 가능 (cpu inference라서 느림)


혹은 [여기](#설치) 설치 후에,

```bash
python gradio-app.py
```
로 실행 가능함
(물론, bravo, charlie와 같은 nato phonetic_alphabet만 인식함)


## 개요 

데이터 수집을 위해 고프로를 들고 야산을 이동하면서, 

- 여기에는 영지버섯이 있습니다. 
- 여기에는 말똥버섯이 있습니다.
- 여기에는 고사리가 자라고 있습니다.

와 같은 scenario를 보이스로 빠르게 태깅하기 위함
 
만약 '여기에는 말똥버섯이 잘 자라고 있습니다.'와 같이 자연어를 사용한다면, automatic speech recognition의 경우 정확도가 95%를 넘기가 힘듬

그러나 아래 나토에서 사용하는 26가지 알파벳을 사용하면, 오직 26개 단어만 학습하면 됨

```python
phonetic_alphabet = {
    "a": "alpha",
    "b": "bravo",
    "c": "charlie",
    "d": "delta",
    "e": "echo",
    "f": "foxtrot",
    "g": "golf",
    "h": "hotel",
    "i": "india",
    "j": "juliet",
    "k": "kilo",
    "l": "lima",
    "m": "mike",
    "n": "november",
    "o": "oscar",
    "p": "papa",
    "q": "quebec",
    "r": "romeo",
    "s": "sierra",
    "t": "tango",
    "u": "uniform",
    "v": "victor",
    "w": "whiskey",
    "x": "x-ray",
    "y": "yankee",
    "z": "zulu"
}
```

다만, 시나리오 목록에 대응하는 약어를 미리 만들어야함 

ex.
- 고사리가 자라고 있음. Go Sa Ri -> GSR -> Golf Sierra Romeo
- 눈이 덮여있어 바닥이 미끄러움. Snow Slippery -> SS -> Sierra Sierra


## 성능 

zeroshot은 당연히 안됨.
WER(word error rate): 44.67

1시간 finetuning 후, WER는 3 미만이며, word 단위로 100개 중 95개 이상을 맞는다고 볼 수 있음.

[모델 링크](https://huggingface.co/famousdetectiveadrianmonk/whisper-small-nato-phoentic-alphabet)


## 학습 데이터 세트 구성 
1. 아무렇거나 4~10 길이의 문자열을 구성함.
ex. dafadf

2.  phonetic alpahbet으로 치환함.
ex. dafadf -> delta alpha foxtrot alpha delta foxtrot

3. tts 모델을 통해 이를 audio로 내보냄. 
그럼 (audio, sentence)로 구성된 샘플 1개가 생성되는 것 

이때, 다양한 voice style로 생성하기 위해, librispeech_asr에서 뽑은 10초 길이의 오디오를 style로 이용함


[데이터세트 링크](https://huggingface.co/datasets/famousdetectiveadrianmonk/nato-phoentic-alphabet-voice)


##  설치

justfile 사용시,

```bash
just install
```

수동 설치시, 

```bash
conda env list | grep  $PWD/venv || conda create -y --prefix $PWD/venv python=3.11 pip ipykernel -q
conda activate $PWD/venv
pip install -U -r requirements.txt    
```

## 기타

학습 코드는 대부분 [transformer 리포](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition)
에서 copy paste함. 이해 못한 부분도 많음
