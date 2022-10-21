## Introduction
1. Simple implementation of JETS (End-To-End) using FastSpeech2, HiFi-GAN, VITS, Conformer open source and fast learning using Korean dataset (KSS).
2. In Adversarial Training, Discriminator uses the same modules used in VITS.
3. Add blank token inside Text Sequence for effective alignment learning.
4. If the l1 reconstructure loss (only log mel magnitude) proposed by HiFi-GAN in this repository is used as it is, an issue occurs in adversarial loss. Therefore, we replaced the log stft magnitude with the stft loss in which the l1 norm is calculated together.
5. For extensibility, VITS Normalizing Flows (CouplingLayer) is used instead of Decoder in the existing FastSpeech2 structure. Therefore, the Posterior Encoder is also used. (Quality improvement, Voice Conversion purpose)
6. The existing Posterior Encoder uses Linear Spectrogram as input, but in this repository, Mel Spectrogram is used.
7. In the existing open source, learning proceeds with MFA-based preprocessing, but in this repository, training data is fed from data_utils.py in order to proceed with alignment learning-based learning and to prevent disk capacity problems that may occur due to preprocessing.
8. It is okay to proceed with the conda environment, but only the docker environment is provided in this repository. It is assumed that docker, nvidia-docker is installed on ubuntu by default.
9. Depending on the type of GPU or CUDA, it may be necessary to modify the torch image at the top of the Dockerfile.
10. In the preprocessing stage, only the process of extracting transcripts and stats necessary for learning is included.
11. No other preprocessing steps are required.
12. More powerful than the previous repository VAEJETS and reduced training time.
13. Because it is based on End-To-End & Adversarial training, it requires a lot of learning to produce high-quality audio.
## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/CVAEJETS/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/CVAEJETS/data/dataset`

## Docker build
1. `cd /path/to/the/CVAEJETS`
2. `docker build --tag CVAEJETS:latest .`

## Training
1. `nvidia-docker run -it --name 'CVAEJETS' -v /path/to/CVAEJETS:/home/work/CVAEJETS --ipc=host --privileged CVAEJETS:latest`
2. `cd /home/work/CVAEJETS`
5. `ln -s /home/work/CVAEJETS/data/dataset/kss`
6. `python preprocess.py ./config/kss/preprocess.yaml`
7. `python train.py -p ./config/kss/preprocess.yaml -m ./config/kss/model.yaml -t ./config/kss/train.yaml`
8. `python train.py --restore_step <checkpoint step number> -p ./config/kss/preprocess.yaml -m ./config/kss/model.yaml -t ./config/kss/train.yaml`
9. arguments
  * -p : preprocess config path
  * -m : model config path
  * -t : train config path
10. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses
![CVAEJETS-tensorboard-losses1](https://user-images.githubusercontent.com/69423543/185771913-20621fca-c0fb-4e41-93f4-905e2ffaa13e.png)
![CVAEJETS-tensorboard-losses2](https://user-images.githubusercontent.com/69423543/185771915-65a16463-91c5-4030-ad46-379cc420de1a.png)


## Tensorboard Stats
![CVAEJETS-tensorboard-stats](https://user-images.githubusercontent.com/69423543/185771918-f14d33d2-e2f3-4bfe-bd66-a7ac9523edac.png)


## Reference
1. [VAENAR-TTS: Variational Auto-Encoder based Non-AutoRegressive Text-to-Speech Synthesis](https://arxiv.org/abs/2107.03298)
2. [JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech](https://arxiv.org/abs/2203.16852)
3. [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS)
4. [Comprehensive-E2E-TTS](https://github.com/keonlee9420/Comprehensive-E2E-TTS)
5. [Conformer](https://github.com/sooftware/conformer) - [paper](https://arxiv.org/abs/2005.08100)
6. [FastSpeech2](https://github.com/ming024/FastSpeech2)
7. [HiFi-GAN](https://github.com/jik876/hifi-gan)
8. [VAEJETS](https://github.com/choiHkk/VAEJETS)
9. [VITS](https://github.com/jaywalnut310/vits)
