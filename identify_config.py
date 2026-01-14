from utils.config_loader import load_config
import json

cfg = load_config()
print(json.dumps(cfg['bot'], indent=4))
