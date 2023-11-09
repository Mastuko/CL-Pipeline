import torch


def check_same_shape(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Raises an error if predictions and targets do not have same dimensions
    """
    if predictions.shape != targets.shape:
        raise RuntimeError(
            "Predictions and Targets are expected to have same dimensions"
        )


def convert_to_tensor(obj):
    """
    Converts to Tensor if given object is not a Tensor.
    """
    if not isinstance(obj, torch.Tensor):
        obj = torch.Tensor(obj)
    return obj


def hyper_wrapfun(**kwargs):
    wrap_dict = {}
    for key in kwargs.keys():
        if isinstance(kwargs[key], str):
            wrap_dict[(str(key),"str")] = [kwargs[key]]
        elif isinstance(kwargs[key], int):
            wrap_dict[(str(key),"int")] = [kwargs[key]+i for i in range(-int(kwargs[key]), int(kwargs[key]))]
        elif isinstance(kwargs[key], float):
            wrap_dict[(str(key),"float")] = [kwargs[key]-0.5*kwargs[key], kwargs[key]+0.5*kwargs[key]]
        else:
            wrap_dict[(str(key),"obj")] = [kwargs[key]]
    return wrap_dict