# PHA
The official implementation of the paper "Prototype-based HyperAdapter for Sample-Efficient Multi-task Tuning"

Acknowledgments: The implementations of the codebase are from the [HyperDecoders](https://github.com/allenai/hyperdecoders/tree/main) repository. Huge thanks to the contributors of those amazing repositories!

# Training
To run our model in a multi-task setting:

`python finetune_trainer.py ./configs/experiments/glue_t5_base.json`
