METRIC_REGISTRY = {}

def register_metric(name):
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator
