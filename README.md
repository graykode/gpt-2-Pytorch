## **gpt-2-Pytorch**

<p align="center"><img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

This repository is only sample implementation of gpt-2 in **Pytorch**. The original repertoire is [openai/gpt-2](https://github.com/openai/gpt-2). Also You can Read Paper about gpt-2, ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). To Understand more detail concept, I recommend papers about Transformer Model.

- [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
- [Improving Language Understanding by Generative Pre-Training(2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- See [OpenAI Blog](https://blog.openai.com/better-language-models/) about gpt-2 and Paper



## Quick Start

To generate unconditional samples from the small model. But in first Commit about BackBone Code, We can't run only this code.. :(

```shell
$ python generate_unconditional_samples.py | tee samples
```



## TODO

- Convert Model tensorflow to Pytorch in [openai/gpt-2](https://github.com/openai/gpt-2) to running.
- Implement Training Option using gpt-2 Model.
- Implement more function using gpt-2 Model.



## About License

Preparing by original repository. I also wait there is a license in place  

- Issuses about License
  - https://github.com/openai/gpt-2/issues/10
  - https://github.com/openai/gpt-2/issues/37