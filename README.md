## **GPT2-Pytorch with Text-Generator**

<p align="center"><img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

**Better Language Models and Their Implications**

> Our model, called GPT-2 (a successor to [GPT](https://blog.openai.com/language-unsupervised/)), was trained simply to predict the next word in 40GB of Internet text. Due to our concerns about malicious applications of the technology, we are not releasing the trained model. As an experiment in responsible disclosure, we are instead releasing a much [smaller model](https://github.com/openai/gpt-2) for researchers to experiment with, as well as a [technical paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). from [openAI Blog](https://blog.openai.com/better-language-models/)

This repository is simple implementation GPT-2 about **text-generator** in **Pytorch** with **compress code**

- The original repertoire is [openai/gpt-2](https://github.com/openai/gpt-2). Also You can Read Paper about gpt-2, ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). To Understand more detail concept, I recommend papers about Transformer Model.
- Good implementation GPT-2 in Pytorch which I referred to, [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT), You can see more detail implementation in huggingface repository.

- Transformer(Self-Attention) Paper : [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
- First OpenAi-GPT Paper : [Improving Language Understanding by Generative Pre-Training(2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- See [OpenAI Blog](https://blog.openai.com/better-language-models/) about GPT-2 and Paper



## Quick Start

1. download GPT2 pre-trained model in Pytorch which huggingface/pytorch-pretrained-BERT already made! (Thanks for sharing! it's help my problem transferring tensorflow(ckpt) file to Pytorch Model!)
```shell
$ git clone https://github.com/graykode/gpt-2-Pytorch && cd gpt-2-Pytorch
# download huggingface's pytorch model 
$ curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
# setup requirements, if using mac os, then run additional setup as descibed below
$ pip install -r requirements.txt
```


2. Now, You can run like this.

- Text from Book 1984, George Orwell

```shell
$ python main.py --text "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
```

3. Also You can Quick Starting in [Google Colab](https://colab.research.google.com/github/graykode/gpt-2-Pytorch/blob/master/GPT2_Pytorch.ipynb)



## Option

- `--text` : sentence to begin with.
- `--quiet` : not print all of the extraneous stuff like the "================"
- `--nsamples` : number of sample sampled in batch when multinomial function use
- `--unconditional` : If true, unconditional generation.
- `--batch_size` : number of batch size
- `--length` : sentence length (< number of context)
- `--temperature`:  the thermodynamic temperature in distribution `(default 0.7)`
- `--top_k`  : Returns the top k largest elements of the given input tensor along a given dimension. `(default 40)`

See more detail option about `temperature` and `top_k` in [here](https://github.com/openai/gpt-2#gpt-2-samples)



## Dependencies

- Pytorch 0.41+
- regex 2017.4.5

### Mac OS Setup
```shell
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install torch tqdm
$ brew install libomp
$ export LC_ALL=en_US.UTF-8
$ export LANG=en_US.UTF-8
$ pip install -r requirements.txt
```

## Author

- Tae Hwan Jung(Jeff Jung) @graykode
- Author Email : [nlkey2022@gmail.com](mailto:nlkey2022@gmail.com)



## License

- OpenAi/GPT2 follow MIT license, huggingface/pytorch-pretrained-BERT is Apache license. 
- I follow MIT license with original GPT2 repository



## Acknowledgement

[Jeff Wu(@WuTheFWasThat)](https://github.com/WuTheFWasThat), [Thomas Wolf(@thomwolf)](https://github.com/thomwolf) for allowing referring code.