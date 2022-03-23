import numpy as np

fixed_params = {
    "max_tokens": 100,
    "best_of": 1
}

tunable_params = {
    "temperature": [0.0, 0.25, 0.5],
    "top_p": [0.7, 0.85, 1.0],
    "frequency_penalty": [0.0, 0.1, 0.2],
    "presence_penalty": [0.0, 0.1, 0.2]
}

def get_max_combinations(params):
    i = 1
    for _, values in params.items():
        i *= len(values)
    return i

def contains_object(arr, b):
    for a in arr:
        is_same = True
        for key in a.keys():
            if a[key] != b[key]:
                is_same = False
                break
        if is_same:
            return True
    return False

def get_random_param_options(n=5):
    max_N = get_max_combinations(tunable_params)
    assert n <= max_N, f"n too large, max options is {max_N}"
    param_options = []
    i = 0
    while i < n:
        params = {}
        for key, values in tunable_params.items():
            params[key] = np.random.choice(values, 1)[0]
        if not contains_object(param_options, params):
            param_options.append(params)
            i += 1
    return param_options