# PHA
The official implementation of the paper "Prototype-based HyperAdapter for Sample-Efficient Multi-task Tuning". [Arxiv]

Acknowledgments: The implementations of the codebase are from the [HyperDecoders](https://github.com/allenai/hyperdecoders/tree/main) repository. Huge thanks to the contributors of the amazing repository!

# Installation

    cd PHA
    pip install -r requirements.txt

# Usage
To run our model in the full-data multi-task setting:

    python finetune_trainer.py ./configs/experiments/glue_t5_base.json

To reproduce the sample efficiency results from Figure 2 of our paper, you would run:

For different quantities:

    # Number of samples in each task：100
    python finetune_trainer.py ./configs/experiments/glue_t5_base_100.json

    # Number of samples in each task：500
    python finetune_trainer.py ./configs/experiments/glue_t5_base_500.json
    
    # Number of samples in each task：1000
    python finetune_trainer.py ./configs/experiments/glue_t5_base_1000.json
    
    # Number of samples in each task：2000
    python finetune_trainer.py ./configs/experiments/glue_t5_base_2000.json
    
    # Number of samples in each task：4000
    python finetune_trainer.py ./configs/experiments/glue_t5_base_4000.json

For different proportions:

    # Number of samples in each task：1%
    python finetune_trainer.py ./configs/experiments/glue_t5_base_001.json

    # Number of samples in each task：3%
    python finetune_trainer.py ./configs/experiments/glue_t5_base_003.json
    
    # Number of samples in each task：5%
    python finetune_trainer.py ./configs/experiments/glue_t5_base_005.json

# Reference
if you find this repository useful, please cite our paper:

    @inproceedings{hao2023hpa,
      title={Prototype-based HyperAdapter for Sample-Efficient Multi-task Tuning},
      author={Hao Zhao and Jie Fu and Zhaofeng He},
      booktitle={Proceedings of EMNLP},
      year={2023}
    }
