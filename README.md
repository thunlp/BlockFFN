# BlockFFN

Source codes for BlockFFN, introduced by the paper: *BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity*.

Links: [[Paper](https://arxiv.org/pdf/2507.08771)] [[Models](https://huggingface.co/SparseLLM)]

1. For codes about the architecture and pre-training process of BlockFFN, see the directory `pretrain`.
2. For codes about the implementation and pre-training process of baseline methods, see the directory `baseline`.
3. For codes about the inference acceleration (i.e., the efficient acceleration kernels) of BlockFFN, see the directory `inference`.

### Citation

If you find our work useful for your research, please kindly cite our paper as follows:

```
@article{song2025blockffn,
      title={{BlockFFN}: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity}, 
      author={Chenyang Song and Weilin Zhao and Xu Han and Chaojun Xiao and Yingfa Chen and Yuxuan Li and Zhiyuan Liu and Maosong Sun},
      journal={arXiv preprint arXiv:2507.08771},
      year={2025},
      url={https://arxiv.org/pdf/2507.08771}, 
}
```
