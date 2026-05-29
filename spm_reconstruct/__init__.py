from spm_eval import (rmse, mae, rmse_by_line, rmse_by_region,
                      ssim, psd_rmse, psd_curves,
                      coverage, mean_uncertainty, nll,
                      classifier_auc, classifier_fpr, classifier_fnr,
                      Evaluator, CollectionReport, ComparisonTable)

MODEL_REGISTRY = {}

def get_model(model_id, params=None):
    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_id}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id](params=params)

def _try_import(module_path, class_name, model_id):
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, class_name)
        MODEL_REGISTRY[model_id] = cls
    except ImportError:
        pass

_try_import('.M01_S2_C0_G0_P1.model', 'M01', '(S2, C0, G0, P1)')
_try_import('.M02_S2_C0_G0_P2.model', 'M02', '(S2, C0, G0, P2)')
_try_import('.M27_S6_CT_GC_P1.model', 'M27', '(S6, CT, GC, P1)')
