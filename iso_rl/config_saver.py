# import pathlib
# import ruamel.yaml as yaml
# from datetime import datetime
# import sys
# import torch

# def save_config(config, logdir_path):
#     """
#     Save configuration to YAML file in the specified log directory.
    
#     Args:
#         config: Configuration object (argparse.Namespace or dict)
#         logdir_path: Path to the log directory (string or pathlib.Path)
#     """
    
#     # Convert to pathlib.Path if it's a string
#     logdir = pathlib.Path(logdir_path).expanduser()
    
#     # Ensure the log directory exists
#     logdir.mkdir(parents=True, exist_ok=True)
    
#     # Convert config to dictionary if it's an argparse.Namespace
#     if hasattr(config, '__dict__'):
#         config_dict = vars(config)
#     else:
#         config_dict = config
    
#     # Convert non-serializable objects to strings
#     config_dict_clean = {}
#     for key, value in config_dict.items():
#         if isinstance(value, pathlib.Path):
#             config_dict_clean[key] = str(value)
#         elif callable(value) and hasattr(value, '__name__'):
#             config_dict_clean[key] = value.__name__
#         elif hasattr(value, '__class__') and 'torch' in str(type(value)):
#             config_dict_clean[key] = str(value)
#         else:
#             try:
#                 # Test if the value can be serialized
#                 yaml.safe_dump(value)
#                 config_dict_clean[key] = value
#             except:
#                 config_dict_clean[key] = str(value)
    
#     # Add timestamp
#     config_dict_clean['saved_at'] = datetime.now().isoformat()
    
#     # Save to YAML file
#     yaml_path = logdir / 'config.yaml'
    
#     yaml_handler = yaml.YAML()
#     yaml_handler.default_flow_style = False
#     yaml_handler.indent(mapping=2, sequence=4, offset=2)
    
#     with open(yaml_path, 'w') as f:
#         yaml_handler.dump(config_dict_clean, f)
    
#     print(f"Configuration saved to: {yaml_path}")
#     return yaml_path

import pathlib
import ruamel.yaml as yaml
from datetime import datetime
# import sys
# import torch
import argparse
from typing import Union, Dict, Any

def save_config(config: Union[argparse.Namespace, Dict[str, Any]], logdir_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save configuration to YAML file in the specified log directory.
    
    Args:
        config: Configuration object (argparse.Namespace or dict)
        logdir_path: Path to the log directory (string or pathlib.Path)
        
    Returns:
        Path to the saved config file
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

def load_saved_config(config_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    Load a previously saved config file (YAML format with proper list handling)
    
    Args:
        config_path: Path to the saved config YAML file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = pathlib.Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # List of parameters that should be lists
    list_params = [
        'encoder_kernels', 'decoder_kernels', 'bg_encoder_kernels', 
        'bg_decoder_kernels', 'size', 'grad_heads'
    ]
    
    # Validate and fix list parameters
    for param in list_params:
        if param in config:
            if not isinstance(config[param], list):
                print(f"Warning: {param} is not a list, got {type(config[param])}: {config[param]}")
                # Try to convert if it's a string representation
                if isinstance(config[param], str):
                    try:
                        # Handle string representations like "[4, 4, 4, 4]"
                        if config[param].startswith('[') and config[param].endswith(']'):
                            config[param] = eval(config[param])
                        else:
                            # Handle space or comma separated values
                            config[param] = [int(x) for x in config[param].replace(',', ' ').split()]
                    except Exception as e:
                        print(f"Could not convert {param} to list: {e}")
                        pass
            else:
                print(f"✓ {param}: {config[param]} (loaded as list)")
    
    # Remove the timestamp from config if present (since it's metadata, not a parameter)
    if 'saved_at' in config:
        print(f"Config was saved at: {config['saved_at']}")
        del config['saved_at']
    
    return config

def convert_config_to_namespace(config_dict: Dict[str, Any]) -> argparse.Namespace:
    """
    Convert config dictionary to argparse.Namespace for easier attribute access
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        argparse.Namespace object
    """
    return argparse.Namespace(**config_dict)

def validate_config_lists(config: Union[argparse.Namespace, Dict[str, Any]]) -> None:
    """
    Validate that list parameters in config are properly formatted
    
    Args:
        config: Configuration object (argparse.Namespace or dict)
    """
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    
    list_params = [
        'encoder_kernels', 'decoder_kernels', 'bg_encoder_kernels', 
        'bg_decoder_kernels', 'size', 'grad_heads'
    ]
    
    print("Validating list parameters:")
    for param in list_params:
        if param in config_dict:
            value = config_dict[param]
            is_list = isinstance(value, list)
            print(f"  {param}: {value} (type: {type(value).__name__}, is_list: {is_list})")
            
            if not is_list:
                print(f"  ⚠️  WARNING: {param} should be a list!")
            else:
                print(f"  ✓ {param} is properly formatted as list")

# # Example usage and testing
# if __name__ == "__main__":
#     # Test the functions
#     import tempfile
    
#     # Create a test config
#     test_config = argparse.Namespace(
#         encoder_kernels=[4, 4, 4, 4],
#         decoder_kernels=[5, 5, 6, 6],
#         size=[64, 64],
#         act='ELU',
#         seed=123,
#         steps=50000
#     )
    
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # Test saving
#         saved_path = save_config(test_config, tmpdir)
        
#         # Test loading
#         loaded_config = load_saved_config(saved_path)
        
#         # Test conversion to namespace
#         loaded_namespace = convert_config_to_namespace(loaded_config)
        
#         # Test validation
#         validate_config_lists(loaded_namespace)
        
#         print(f"\nOriginal encoder_kernels: {test_config.encoder_kernels}")
#         print(f"Loaded encoder_kernels: {loaded_namespace.encoder_kernels}")
#         print(f"Types match: {type(test_config.encoder_kernels) == type(loaded_namespace.encoder_kernels)}")
#         print(f"Values match: {test_config.encoder_kernels == loaded_namespace.encoder_kernels}")