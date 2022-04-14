import json
import openai
from hyperparams import get_random_param_options
from setup import setup
from utility import get_base_model_name, get_final_model_name

def get_base_models():
    """
    Fetches all fine-tuned base-models from OpenAI,
    and returns a list of models' hyperparams.

    Parameters
    ----------

    Returns
    ----------
    models (list[dict[str, Any]]): List of models with shape [{ id: model_openai_id, params: model_hyper_params }]
    """
    tune_param_options = get_random_param_options(type="FINE_TUNE", get_all=True)
    # Base model names derived from hyperparameter options
    base_model_names = [
        (get_base_model_name(params), params)
        for params in tune_param_options
    ]
    
    fine_tuned_models = openai.FineTune.list()["data"]
    models = []
    # Check that a fine tuned model can be found for all given parameter combinations
    for (name, params) in base_model_names:
        # Check that name matches with a fine-tuned model
        model = next((m for m in fine_tuned_models if name in m["fine_tuned_model"]), None)
        #assert model is not None, f"Model {name} was not found in OpenAI. You need to fine tune the model first."
        models.append({
            "params": params,
            "model_id": model["fine_tuned_model"]
        })
    return models
    

def create_models():
    """
    Takes base models saved to OpenAI and creates final models with fixed request params.
    Creates a model for each possible request-parameter combination.
    Total number of returned models will be n*m, where n is the number of base models
    and m is the number of possible request-param combinations.

    Parameters
    ----------
    base_models (list[str]): List of model_names (saved to OpenAI) of the models from which to create final models.


    Returns
    ----------
    models (list[Any]): list of the created models
    """
    fine_tuned_models = get_base_models()
    params = get_random_param_options('REQUEST', get_all=True)
    models = {}
    for model in fine_tuned_models:
        for p in params:
            model_id = model["model_id"]
            fine_tune_params = model["params"]
            name = get_final_model_name(fine_tune_params, p)
            models[model_id] = {
                "name": name,
                "params": p
            }
    with open("models.json", "w") as f:
        json.dump(models, f)

setup()
create_models()
