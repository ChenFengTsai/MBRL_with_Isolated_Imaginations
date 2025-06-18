import pathlib
import ruamel.yaml as yaml
from datetime import datetime
import sys
import torch

def save_config(config, logdir_path):
    """
    Save configuration to YAML file in the specified log directory.
    
    Args:
        config: Configuration object (argparse.Namespace or dict)
        logdir_path: Path to the log directory (string or pathlib.Path)
    """
    
    # Convert to pathlib.Path if it's a string
    logdir = pathlib.Path(logdir_path).expanduser()
    
    # Ensure the log directory exists
    logdir.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary if it's an argparse.Namespace
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    
    # Convert non-serializable objects to strings
    config_dict_clean = {}
    for key, value in config_dict.items():
        if isinstance(value, pathlib.Path):
            config_dict_clean[key] = str(value)
        elif callable(value) and hasattr(value, '__name__'):
            config_dict_clean[key] = value.__name__
        elif hasattr(value, '__class__') and 'torch' in str(type(value)):
            config_dict_clean[key] = str(value)
        else:
            try:
                # Test if the value can be serialized
                yaml.safe_dump(value)
                config_dict_clean[key] = value
            except:
                config_dict_clean[key] = str(value)
    
    # Add timestamp
    config_dict_clean['saved_at'] = datetime.now().isoformat()
    
    # Save to YAML file
    yaml_path = logdir / 'config.yaml'
    
    yaml_handler = yaml.YAML()
    yaml_handler.default_flow_style = False
    yaml_handler.indent(mapping=2, sequence=4, offset=2)
    
    with open(yaml_path, 'w') as f:
        yaml_handler.dump(config_dict_clean, f)
    
    print(f"Configuration saved to: {yaml_path}")
    return yaml_path