def get_prompt(filename):
    with open(filename, 'r') as f:
        return f.read()