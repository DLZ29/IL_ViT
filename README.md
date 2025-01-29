# IL_ViT
This is a demo to test ViT-based foundation models for image-navigation tasks, using immitation learning with Habitat Sim.

## Environment Configuration
Ensure that Habitat-sim and Habitat-lab are installed properly. For datasets, please refer to [GibsonEnv](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md) and [PointNav (Task Dataset)](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2019)

## Data Generation
```bash
python collect_IL_data.py --ep-per-env 50 --num-procs 4 --split train --data-dir /your-path-to-save-data
```
This will generate data for immitation learning. Settings can be customed in `collect_IL_data.py`.

## Training
```bash
python train_bc.py --config configs/vgm.yaml --stop --gpu 0
```
   By default we use a simplified OVRL model with ViT and compression layer. By comparison to a single ViT or ResNet, rewrite `policy = ` part
## Evaluation
Load your trained model and run:
```
python validation.py
```
If the agent reaches <1m distance to the goal, the current task episode is marked success. Continous stuck or overtime is marked unsuccess. 
Current and average SR, SPL, DTS are displayed for review.
The output of the stop action `0` in the simulator is replaced by the forward action `1` to avoid the extreme impact of data sparsity on imitation learning. After all, this is just a demo to test the foundation model.
