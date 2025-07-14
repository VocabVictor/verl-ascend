"""Local logger implementation for VERL."""

import logging
from typing import Dict, Any

class LocalLogger:
    """Simple local logger for VERL training."""
    
    def __init__(self, name: str = "verl", print_to_console: bool = True):
        self.logger = logging.getLogger(name)
        self.print_to_console = print_to_console
        
        if print_to_console:
            # Set up console handler if not already present
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
    def log(self, data: Dict[str, Any], step: int = None):
        """Log training metrics."""
        if step is not None:
            self.logger.info(f"Step {step}: {data}")
        else:
            self.logger.info(f"Metrics: {data}")
            
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
        
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)