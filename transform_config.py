from typing import Dict, Any, Optional, Union
import yaml
import os


class TransformConfig:
    """Configuration class for weight transformations."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, method: Optional[str] = None):
        """Initialize the configuration.
        
        Args:
            config_dict: Dictionary containing the configuration.
            method: Transformation method to use. Only used if config_dict is None.
        """
        if config_dict is None:
            if method is None:
                method = 'origin'
            self.config_dict = {'method': method}
        else:
            self.config_dict = config_dict
            
    @property
    def method(self) -> str:
        """Get the transformation method."""
        return self.config_dict.get('method', 'origin')
    
    @method.setter
    def method(self, value: str) -> None:
        """Set the transformation method."""
        self.config_dict['method'] = value
        
    def get_method_params(self) -> Dict[str, Any]:
        """Get the parameters for the current method."""
        return self.config_dict.get(self.method, {})
    
    def set_method_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters for the current method."""
        self.config_dict[self.method] = params
        
    def get_param(self, param_name: str, default: Any = None) -> Any:
        """Get a parameter for the current method."""
        method_params = self.get_method_params()
        return method_params.get(param_name, default)
    
    def set_param(self, param_name: str, value: Any) -> None:
        """Set a parameter for the current method."""
        if self.method not in self.config_dict:
            self.config_dict[self.method] = {}
        self.config_dict[self.method][param_name] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return self.config_dict
    
    @classmethod
    def from_file(cls, file_path: str) -> 'TransformConfig':
        """Load a configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_preset(cls, preset_name: str, config_dir: str = 'config/transform') -> 'TransformConfig':
        """Load a preset configuration."""
        file_path = os.path.join(config_dir, f'{preset_name}.yaml')
        return cls.from_file(file_path)
    
    def save_to_file(self, file_path: str) -> None:
        """Save the configuration to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.config_dict, f, default_flow_style=False)
            
    def __getitem__(self, key: str) -> Any:
        """Get a parameter or section from the configuration."""
        return self.config_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a parameter or section in the configuration."""
        self.config_dict[key] = value
        
    def __contains__(self, key: str) -> bool:
        """Check if a parameter or section exists in the configuration."""
        return key in self.config_dict


def get_transform_config(config) -> Union[TransformConfig, str, Dict[str, Any]]:
    """Get a transform configuration from various input types.
    
    Args:
        config: Can be:
            - A TransformConfig object (returned as is)
            - A string (treated as method name or preset name)
            - A dictionary (converted to TransformConfig)
            - None (returns default config)
            
    Returns:
        A TransformConfig object, or the input if it's a string or dict.
    """
    return config


# Example usage:
# config = TransformConfig.from_preset('aggressive')
# print(config.method)  # 'threshold_and_scale'
# print(config.get_param('min_val'))  # -2.0
# 
# config.method = 'binary'
# config.set_param('top_percent', 50)
# print(config.to_dict()) 