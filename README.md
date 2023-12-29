# Re-implementation of Let's Verify Step by Step

This project is a re-implementation of OpenAI's "Let's Verify Step by Step" paper, aimed at running on low-resource hardware. The paper is accessible [here](https://arxiv.org/pdf/2305.20050.pdf).

## Overview

The project involves finetuning a language model on a math corpus, consisting of math problems, solutions, and general math texts. This finetuned model is then trained to become a Process-supervised Reward Model (PRM), which serves as a reward model in a reinforcement learning context. Specifically, it's used to train models like GPT-2 to provide more accurate solutions to math problems. The PRM incorporates a classifier head replacing the usual logprob output layer, to classify solutions as negative, neutral, or positive.

## Environment

- Nvidia RTX 3090
- Ubuntu OS
- 48 GB RAM (for loading weights)

## Steps

1. **Pre-Training LLaMA 2 on Math Corpus**: Begin by pre-training the LLaMA 2 model on a math corpus using causal language modeling.
2. **Creating PRM Model**: Take the pre-trained LLaMA 2 and finetune it using OpenAI's PRM800k dataset, which contains math problems with step-by-step solutions.
3. **Classifier Head Integration**: Integrate a classifier head into the PRM model, enabling it to output a negative, neutral, or positive classification of a math solution.
4. **Reinforcement Learning**: Utilize the PRM to conduct reinforcement learning, improving GPT-2's ability to output better math solutions.

## Usage
TODO

## License

MIT

