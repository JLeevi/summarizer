# TODO

## Pipeline for training

### Prepare data
    - Download datasets
    - Choose how many samples to draw from each dataset
    - Script to automate buildling a dataset, params

        `n`=how many samples

        `distribution`=propability distrib on dataset indices

        i.e. 500 samples from 4 datasets:

        `n=100`

        `distribution={0: 0.2, 1, 0.4, 2: 0.2, 3: 0.2}`
    
    - Create JSONL-file and use prep using `fine_tunes.prepare_data`

### Build pipeline for testing hyperparams
