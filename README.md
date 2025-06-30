# TADFormer : Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning
## Run TADFormer

Running TADFormer code:

**Run the code**
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 \
main.py --cfg configs/TADFormer/[config_name].yaml \
--pascal [pascal_dataset] --tasks semseg,normals,sal,human_parts \
--batch-size 32 --ckpt-freq=20 --epoch=300 --resume-backbone [Pretrained Swin Transformer .pth path] \
--disable_wandb
```

**Eval **
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 \
main.py --cfg configs/TADFormer/[config_name].yaml \
--pascal [pascal_dataset] --tasks semseg,normals,sal,human_parts \   
--batch-size 32 --ckpt-freq=20 --epoch=300 --resume [.pth path] \
--eval \
--disable_wandb
```



