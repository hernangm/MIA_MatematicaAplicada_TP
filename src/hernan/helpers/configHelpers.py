def extract(data: dict, keys: list) -> tuple:
    values = tuple(data[key] for key in keys)
    return values[0] if len(values) == 1 else values