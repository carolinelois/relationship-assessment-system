import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self):
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
        env = os.getenv("ENVIRONMENT", "development")

        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)

            # Load environment-specific config if it exists
            env_config_path = Path(config_path).parent / f"config.{env}.yaml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                    self._deep_update(self._config, env_config)

            # Override with environment variables
            self._override_from_env(self._config)

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        for key, value in update_dict.items():
            if (
                key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _override_from_env(self, config: Dict[str, Any], prefix: str = ""):
        for key, value in config.items():
            env_key = f"{prefix}_{key}".upper().strip("_")
            
            if isinstance(value, dict):
                self._override_from_env(value, env_key)
            else:
                env_value = os.getenv(env_key)
                if env_value is not None:
                    # Convert environment variable to appropriate type
                    if isinstance(value, bool):
                        config[key] = env_value.lower() in ('true', '1', 'yes')
                    elif isinstance(value, int):
                        config[key] = int(env_value)
                    elif isinstance(value, float):
                        config[key] = float(env_value)
                    else:
                        config[key] = env_value

    def get_config(self) -> Dict[str, Any]:
        return self._config

    def get_value(self, *keys: str, default: Any = None) -> Any:
        current = self._config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default
        return current

config = ConfigLoader()