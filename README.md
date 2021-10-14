
# Implementation of StyleGan: A Style-Based Generator Architecture for Generative Adversarial Networks

This repository contains the implementation of all blocks of StyleGAN paper, "A Style-Based Generator Architecture for Generative Adversarial Networks".


The Style Generative Adversarial Network, or StyleGAN for short, is an extension to the GAN architecture that proposes large changes to the generator model, including the use of a mapping network to map points in latent space to an intermediate latent space, the use of the intermediate latent space to control style at each point in the generator model, and the introduction to noise as a source of variation at each point in the generator model.

The resulting model is capable not only of generating impressively photorealistic high-quality photos of faces, but also offers control over the style of the generated image at different levels of detail through varying the style vectors and noise.

Some of the StyleGAN examples:


![stylegan-teaser](https://user-images.githubusercontent.com/16369846/137173555-fa513ae1-a6cc-4237-9b10-0d4c323837fe.png)


Paper: https://arxiv.org/abs/1812.04948
Useful links:
* https://jonathan-hui.medium.com/gan-stylegan-stylegan2-479bdf256299
* https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/
## Repository:

The repositories contain the following files.
* styleGan_blocks(Directory)
	* This directory contains all the blocks of StyleGAN.
		* Adaptive_instance_network.py
		* mapping_network.py
		* noise_injection.py
		* progressive_growing.py
* helper_functions. py
	* This file contains small functions that were use in other files.
*  MicroStyleGan. py
	* This file utlized all StyleGAN blocks and build an micro StyleGan.
* test_styleGan.py
	* This files tests our implementation.

### Remaining work
* Implement end-to-end styleGan
* Add training and inference code
