# car attribute

## Get Started

多属性分类模型修改，给模型添加neck，使每个头输出效果最优化。

1. `cd` to folder where you want to download this repo

2. Run `git clone 

3. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - [yacs](https://github.com/rbgirshick/yacs)

4. Prepare dataset


5. Prepare pretrained model if you don't have

    （1）Resnet

    ```python
    from torchvision import models
    models.resnet50(pretrained=True)
    ```
    （2）Senet

    ```python
    import torch.utils.model_zoo as model_zoo
    model_zoo.load_url('the pth you want to download (specific urls are listed in  ./modeling/backbones/senet.py)')
    ```
    Then it will automatically download model in `~/.torch/models/`, you should set this path in `config/defaults.py` for all training or set in every single training config file in `configs/` or set in every single command.

    （3）Load your self-trained model

    If you want to continue your train process based on your self-trained model, you can change the configuration `PRETRAIN_CHOICE` from 'imagenet' to 'self' and set the `PRETRAIN_PATH` to your self-trained model. 

6. If you want to know the detailed configurations and their meaning, please refer to `config/defaults.py`. If you want to set your own parameters, you can follow our method: create a new yml file, then set your own parameters.  Add `--config_file='config/your yml file'` int the commands described below, then our code will merge your configuration. automatically.


## Date
用以下格式生成train.txt 和 val.txt即可。对应txt路径写出config文件夹对应配置文件中。

```python
#图片路径 属性1对应索引 属性2对应索引 属性3对应索引（ps：中间都存在一个空格，图片路径中不可以有空格）

#eg:
/media/中东数据/国家分类数据/middle_att3_0419/Lebanon/white/red/Lebanon_white_00016758.jpg 3 4 3
/media/中东数据/国家分类数据/middle_att3_0419/Lebanon/white/red/Lebanon_white_00018923.jpg 3 4 3


```



## Train

```bash
python3 tools/train.py --config_file='config/middle_att.yml' MODEL.DEVICE_ID "('your device id'，只有一张显卡可以省去)" OUTPUT_DIR "output/middle_3att"
```

## Test

Please replace the data path of the model and set the `PRETRAIN_CHOICE` as 'self' to avoid time consuming on loading ImageNet pretrained model.

```bash
python3 tools/test.py --config_file='configs/your.yml' MODEL.DEVICE_ID "('your device id')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
```

