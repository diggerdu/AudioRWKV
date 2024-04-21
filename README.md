## Environments
The codebase is developed with pytorch == 1.8.1, torch-lightning == 1.5.9
Install requirements as follows:
```
pip install -r requirements.txt
```

## Training 
First config the model parameters in config.py

For tiny model
```
img_size = (992,64)
patch_size = 16
rwkv_embed_dim = 192
rwkv_num_heads = 3
rwkv_dropout_rates = 0.0
```

for small model
```
img_size = (992,64)
patch_size = 16
rwkv_embed_dim = 384
rwkv_num_heads = 6
rwkv_dropout_rates = 0.3
rwkv_init_values = 1e-5
```

For base model
```
img_size = (992,64)
patch_size = 16
rwkv_embed_dim = 768
rwkv_num_heads = 12
rwkv_dropout_rates = 0.5
rwkv_post_norm = True
rwkv_init_values = 1e-5
```

Start Training:
```
python main.py train
```

## Results
AudioSet: 
|     | RWKV tiny | RWKV small | RWKV base |
|-----|------|-------|------|
| mAP | 40.1  |      |      |
|      |       |      |      |
|      |       |      |      |

