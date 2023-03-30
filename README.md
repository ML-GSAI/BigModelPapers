# BigModelPapers

A collection of some large model papers.

# Paper list

## Text-to-image model

### Diffusion model

* **DALL-E-2**: Hierarchical Text-Conditional Image Generation with CLIP Latents

  DALL-E-2 is a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image
conditioned on the image embedding. [paper](https://arxiv.org/abs/2204.06125)

* **Stable Diffusion**: Hierarchical Text-Conditional Image Generation with CLIP Latents

  Latent diffusion model (Base version of Stable Diffusion) use diffusion model to model latent space of image, and introduces
cross-attention layers into the model architecture to enable conditional generation. [paper](https://arxiv.org/abs/2112.10752)

* **Imagen**: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

    Imagen builds on the power of large transformer language models(e.g. T5) in understanding text and uses cascaded diffusion models to
generate high-fidelity image. [paper](https://arxiv.org/abs/2205.11487)

### Autoregressive models
* **DALL-E**: Zero-Shot Text-to-Image Generation
    
    Use transformer to autoregressively model text and image tokens as a single stream of data. [paper](https://arxiv.org/abs/2102.12092)

* **Parti**: Scaling Autoregressive Models for Content-Rich Text-to-Image Generation

  Parti treats text-to-image generation as a sequence-to-sequence modeling problem, akin to machine translation, with sequences of image tokens as the target outputs rather than text tokens in
another language. [paper](https://arxiv.org/abs/2206.10789)

### Others

* **Muse**: Muse: Text-To-Image Generation via Masked Generative Transformers

  Given the text embedding extracted from a pre-trained large language model (LLM), Muse is trained to predict randomly 
masked image tokens. Compared with diffusion or autoregressive models, Muse is more efficient. [paper](https://arxiv.org/abs/2301.00704)

## Unified generative model
* **UniDiffuser**: UniDiffuser claims that learning diffusion models for marginal, conditional, and joint distributions can
be unified as predicting the noise in the perturbed data, where the perturbation levels (i.e. timesteps)
can be different for different modalities. UniDiffuser is able to perform image, text, text-to-image, image-to-text, and image-text 
pair generation by setting proper timesteps without additional overhead. 
[paper](https://ml.cs.tsinghua.edu.cn/diffusion/unidiffuser.pdf)

## Vision-Language Pre-training model
* **CLIP**: Learning Transferable Visual Models From Natural Language Supervision 

  The pre-training task of CLIP is predicting which caption goes with which image through contrast learning loss. The model transfers 
non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training.
[paper](https://arxiv.org/abs/2103.00020)

* **BEiT 3**: Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks
  BEiT 3 introduce Multiway Transformers for general-purpose model, and use maskd "language" modeling on images, texts and imgae-text 
pairs. [paper](https://arxiv.org/abs/2208.10442)


## Large Language model

* **GPT-3**: Language Models are Few-Shot Learners

  GPT-3 is trained to predict the next word in a sentences. However, model developers and early users demonstrated that it 
had surprising capabilities, like the ability to write convincing essays, create charts and websites from text descriptions, 
generate computer code, and more — all with limited to no supervision. [paper](https://arxiv.org/abs/2005.14165)

