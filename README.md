# README.md

# LESOD

This project provides the code and results for 'LESOD: Lightweight and Efficient Network for RGB-D Salient Object Detection'. You can reproduce our results by following steps.
<hr>

# Environments

```bash
conda create -n lesod python=3.9.18
conda activate lesod
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

# Data Preparation

- Download the RGB-D raw data from [Baidu](https://pan.baidu.com/s/10Y90OXUFoW8yAeRmr5LFnA?pwd=exwj) / [Google Drive](https://drive.google.com/file/d/19HXwGJCtz0QdEDsEbH7cJqTBfD-CEXxX/view?usp=sharing) <br>
- Download the RGB-T raw data from [Baidu](https://pan.baidu.com/s/1eexJSI4a2EGoaYcDkt1B9Q?pwd=i7a2) / [Google Drive](https://drive.google.com/file/d/1hLhn5WV6xh-Q41upXF-bzyVpbszF9hUc/view?usp=sharing) <br>

Note that in the depth maps of the raw data above, the foreground appears white.

# Training & Testin

- Train the MAGNet:
    1. Download the pretrained EdgeNeXt and MobileNetV3 pth from [Baidu](https://pan.baidu.com/s/11bNtCS7HyjnB7Lf3RIbpFg?pwd=bxiw) / [Google Drive](https://drive.google.com/file/d/1mJsIvMjmoOEPrLp5-CxcuFwNk3vAa8E5/view?usp=sharing), and put them under `ckps/`.
    2. Modify the data path in [`train_Net.py`](https://github.com/mingyu6346/LESOD/blob/main/train_Net.py) according to your data path.
    3. Run `python train_Net.py`
- Test the MAGNet:
    1. Modify the `test_path` path in [`test_Net.py`](https://github.com/mingyu6346/LESOD/blob/main/test_Net.py) according to your data path.
    2. Run `python test_Net.py`

# Evaluate tools

[CODToolbox](https://github.com/DengPingFan/CODToolbox) 

# Saliency Maps and Trained Models

| Task | Saliency Maps |  Trained Models |
| --- | --- | --- |
| RGB-D | [Baidu](https://pan.baidu.com/s/1rlxroAixS0hOqUj2Pv6SPw?pwd=4f76) / [Google](https://drive.google.com/file/d/1cLC3uJeNDKig8yFGHdMwqxiur1VSfRdn/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1gvL7mwzejVK0d6jQT101QA?pwd=frag) / [Google](https://drive.google.com/file/d/1nO2DKIbjC_ciRrIwBe7hxSbnYQ042YXT/view?usp=sharing) |
| RGB-T | [Baidu](https://pan.baidu.com/s/1DDc4BJuaGoE8PR8acfmWOA?pwd=e52c) / [Google](https://drive.google.com/file/d/18ssIX1S3yPqLqUKrh9yh2XgCTVWx3fco/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1Iph5-E9nByoiQKfH8oWORQ?pwd=bgyn) / [Google](https://drive.google.com/file/d/1CdoGFAj5en7kdRPSEpPg_sByvVmqACJv/view?usp=sharing) |

# Acknowledgement

The implementation of this project is based on the codebases below. <br>

- [EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt) <br>
- [MobileNetV3](https://arxiv.org/abs/1905.02244) <br>

# Contact

Feel free to contact me if you have any questions: (mingyu6346 at 163 dot com)
