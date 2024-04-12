from omegaconf import DictConfig
from optuna import Trial


def configure_double_poly(cfg: DictConfig, trial: Trial) -> None:
    """Set hyperparemeters for double poly trail
    This allows setting conditional parameters
    Args:
        cfg (DictConfig): hydra config
        trial (Trial): optuna trail object, to suggest hp
    """

    for part in ["part1", "part2"]:
        # iter over part, ie. point compression or time compression
        for name_layers in ["pre_layer", "bet_layer"]:
            nr_pre_layers = trial.params[f"+model.CONTEXT_ENCODER.double_poly.{part}.num_{name_layers}s"]

            dims_layer = []
            for i in range(nr_pre_layers):
                # suggest hidden dimension for layer
                trial.suggest_categorical(
                    f"++model.CONTEXT_ENCODER.double_poly.{part}.dim_{name_layers}_{i}",
                    cfg["hp_params"]["hidden_dim_choices"],
                )
                dims_layer.append(trial.params[f"++model.CONTEXT_ENCODER.double_poly.{part}.dim_{name_layers}_{i}"])

            # set model cfg layers
            cfg["model"]["CONTEXT_ENCODER"]["double_poly"][part][f"{name_layers}s"] = dims_layer
