# finetuning
Demonstrating LLM fine tuning concepts.

# Try it out
This project is packaged with [PDM](https://pdm-project.org/en/latest/).
[Install PDM](https://pdm-project.org/en/latest/#installation), clone this repo, navigate to its root and run `pdm install`.

## Plan
In this project, I would like to start simple by fine tuning GPT2 (because it is so small) to single turn converstations (Q-A-format) on the [Oasst dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) to talk about a single topic, no matter what the user asks.
I chose this task to get my hands dirty with fine tuning (possibly different fine tuning algorithms), and because it might look funny if an LLM always responds in a certain way (see e.g. [Goody2](https://www.goody2.ai/), taking this one step further).

### Progress
I am tracking my progess in `diary/`.
Below are summaries.
- Day 1: Trained first models, did not get around to proper evaluation due to OOM.

### Milestones
- Literature Review
    - I have read some articles on [The Open Assistant Project](https://arxiv.org/pdf/2304.07327), [Fine Tuning GPT-2 with Different Algorithms](https://medium.com/@aalokpatwa/dpo-and-sft-tuning-of-gpt-2-in-pure-pytorch-d025dff6333f), [Fine Tuning GPT-2 on the Alpaca Dataset](https://debuggercafe.com/instruction-tuning-gpt2-on-alpaca-dataset/), and watched a video about [Multi-Head Latent Attention](https://www.youtube.com/watch?v=0VLAoVGf_74).
    - These complement my prior knowledge about LLMs and their training, but from a more practical stand point
    - The video is particularly interesting because while this technique was used to train deepseek R1 from scratch, I imagine you could "speed up" existing MHA models by constructing such a latent representation after training by decomposing the weight matrices.
    This could be an interesting project, or maybe there is already existing research in that direction.
- Basic Fine-Tuning
    - Will be very similar to the [GPT2/Alpaca article](https://debuggercafe.com/instruction-tuning-gpt2-on-alpaca-dataset/).
    One comment I have on that article is that it would be very easy to prompt inject since there is no sanitization of user inputs. 
    But I guess at this model size we are not too concerned with AI alignment at that level.
- Basic hosting
    - Would be cool to make a docker container that automatically serves this model on localhost or similar.