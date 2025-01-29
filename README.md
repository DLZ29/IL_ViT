# IL_ViT
This is a demo to test ViT-based foundation models for image-navigation tasks, using immitation learning with Habitat Sim.

## 环境配置
Ensure that Habitat-sim and Habitat-lab are installed properly. For datasets, please refer to [GibsonEnv](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md) and [PointNav (Task Dataset)](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2019)

## Data Generation
    ```
    python collect_IL_data.py --ep-per-env 50 --num-procs 4 --split train --data-dir /your-path-to-save-data
    ```
    This will generate data for immitation learning. Settings can be customed in `collect_IL_data.py`.

## Training
    ```
   python train_bc.py --config configs/vgm.yaml --stop --gpu 0
    ```
    
## Evaluation
   ```
   python validation.py
   ```
