import json
from mmcls.models import build_classifier

def load_and_init_vrwkv(cfg_file):
    with open(cfg_file) as f:
        cfg_model = json.load(f)
        model = build_classifier(cfg_model)
        model.init_weights()
    return model

if __name__ == '__main__':
    from mmcls_custom import *
    model = load_and_init_vrwkv('cfg_model_tiny.json')
    print('model loading succeed')