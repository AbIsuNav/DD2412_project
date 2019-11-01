# Final project DD2412

This is final project from the course DD2412-Deep Learning, Advanced project.

In this project we reproduce the experiments and conducted a study from the paper:
- Zongsheng Yue, Hongwei Yong, Qian Zhao, Lei Zhang, and Deyu Meng. Variational denoising network: Toward blind noise modeling and removal. (NeurIPS, 2019) [arXiv](https://arxiv.org/pdf/1908.11314v2.pdf)
## Requirements:
* Python 3.7.* 
* Pytorch 1.2.0

## How to run:
### Training: 
For simulated noise:
```bash
python simulation_training.py
```
For benchmark train:
- First obtain the training and validation data files by running:
```bash
python datasets/train_data_sidd.py
```
```bash
python datasets/validation_data_sidd.py
```
- Then train by running:
```bash
python benchmark_training.py
```
this file performs training and validation 
### Testing
For testing the simulation train:
```bash
python Testing_simulation.py
```
## Authors:
- [Abgeiba Isunza Navarro](https://github.com/AbIsuNav)
- [Walid A Jalil](https://github.com/walidjalil)
- [Pratima Rao A](https://github.com/pratima1159)