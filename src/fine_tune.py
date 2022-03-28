from Summarizer import Summarizer
from create_dataset import create_dataset
from hyperparams import get_random_param_options
import openai
from setup import setup

### MODEL FINETUNING

##  - Get combinations of hyperparams
##      - fine_tuning hyp. params
##      - request hyp. params

##  - Get desired prompt-style
##  - Get finetuning dataset

##  - Create finetune job
##  - Save model with params as fstring

##  - Get n text-samples from validation dataset
##  - Use model to summarize text-samples

##  - Save generated summarizations with model params as fstring



def create_summaries():
    # Get n text-samples from validation dataset
    # Use model to summarize text-samples
    # Save summaries with model id / hyperparam string
    pass

def get_model_name(params):
    lr = str(params["learning_rate_multiplier"]).replace('.','')
    prompt_style = params["prompt_style"].name
    return f"model_{prompt_style}_{lr}"

def fine_tune_exists(model_name: str):
    model_name = model_name.lower()
    model_name = model_name.replace('_', '-')
    res = openai.FineTune.list()
    fine_tune_names: list[str] = [f.fine_tuned_model for f in res.data if f.fine_tuned_model != None]
    for name in fine_tune_names:
        if model_name in name:
            return True
    return False

def upload_file(file_path):
    file_name = file_path.split('/')[-1]
    res = openai.File.list()
    f = list(filter(lambda f: f.filename == file_name, res.data))
    if len(f) == 0:
        print(f"Uploading file {file_name}...")
        return openai.File.create(
            file=open(file_path),
            purpose="fine-tune")
    else:
        print(f"File {file_name} already uploaded")
        return f[0]

def create_fine_tuned_model(model_name:str, params: dict, use_validation=False):
    """
    Builds train/test datasets and creates a fine-tune job with given params.
    
    Parameters
    ----------
    model_name (str): Suffix for the created model

    params (dict): Fine-tune parameters, see https://beta.openai.com/docs/api-reference/fine-tunes/create. Must
    also include prompt_style key, which is used to build a dataset of the given format

    use_validation (bool): If true, performs validation for the model with a separate validation dataset
    """
    prompt_style = params.pop("prompt_style")
    train_file = create_dataset(prompt_style=prompt_style)
    train_file = upload_file(train_file)

    test_file = create_dataset(prompt_style=prompt_style, training=False)
    test_file = upload_file(test_file)

    summarizer = Summarizer(model_name)
    print("Staring fine tune job...\n")
    if use_validation:
        res = summarizer.fine_tune(params, train_file.id, test_file.id)
    else:
        res = summarizer.fine_tune(params, train_file.id)
    print(f"Fine tune {res.id} finished \n")

def fine_tune():
    param_options = get_random_param_options(type="FINE_TUNE", get_all=True)
    n = len(param_options)
    for i, params in enumerate(param_options, 1):
        model_name = get_model_name(params)
        print(f"\nFINE TUNE [{i}/{n}]: {model_name}")
        exists = fine_tune_exists(model_name)
        if exists:
            print(f"{model_name} already fine-tuned")
        else:
            use_validation = (i == 1) # only use validation for the first model for now
            create_fine_tuned_model(model_name, params, use_validation)
    
    print("Done!")

setup()
fine_tune()