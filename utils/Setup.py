from typing import Any, Optional, Union
import torch
import logging
import random
import numpy as np
from torch import Tensor

from config import setup_info


def setup(args: Any) -> logging.Logger:
    logger = setup_logging()
    setup_info(args)
    setup_device(args)
    setup_seed(args)
    return logger


def setup_device(args: Any) -> None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args: Any) -> None:
    if not hasattr(args, 'seed'):
        raise ValueError("args must contain a 'seed' attribute")
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging() -> logging.Logger:
    """
    Configure logging for the application.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    return logger


def trans_to_cuda(variable: Union[Tensor, Any]) -> Union[Tensor, Any]:
    """
    Transfer a variable to CUDA if available.
    """
    if not torch.cuda.is_available():
        return variable
        
    device = torch.device("cuda:0")
    try:
        return variable.to(device)
    except (AttributeError, RuntimeError) as e:
        logging.warning(f"Failed to transfer variable to CUDA: {e}")
        return variable
