# R2Gen

This is the implementation of [Generating Radiology Reports via Memory-driven Transformer](https://arxiv.org/pdf/2010.16056.pdf) at EMNLP-2020.
There are changes in it to apply RL methods to This paper model and add some medical components to perform better. Also, it works for the FFA-IR dataset.


This project introduces a novel model for generating medical reports from Fundus Fluorescein Angiography (FFA) images. The model utilizes a Convolutional Neural Network (CNN) as a Visual Extractor to extract visual features from the images. It aligns the visual and textual features of an image and its report using Cross-modal Memory. The encoder-decoder in this model is built upon a standard Transformer. The proposed reinforcement learning (RL) algorithm leverages signals from natural language generation (NLG) metrics, such as BLEU, to guide the cross-modal mappings and generate a comprehensive report.

## Requirements

- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`


It is based on this repository [here](https://github.com/cuhksz-nlp/R2Gen).
