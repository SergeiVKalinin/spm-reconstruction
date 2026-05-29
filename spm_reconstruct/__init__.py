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

# ── 12 core models ────────────────────────────────────────────────
_try_import('.M01_S2_C0_G0_P1.model', 'M01', '(S2, C0, G0, P1)')
_try_import('.M02_S2_C0_G0_P2.model', 'M02', '(S2, C0, G0, P2)')
_try_import('.M07_S2_CT_G0_P1.model', 'M07', '(S2, CT, G0, P1)')
_try_import('.M08_S2_CT_G0_P2.model', 'M08', '(S2, CT, G0, P2)')
_try_import('.M09_S2_CT_GC_P1.model', 'M09', '(S2, CT, GC, P1)')
_try_import('.M10_S2_CT_GC_P2.model', 'M10', '(S2, CT, GC, P2)')
_try_import('.M19_S6_C0_G0_P1.model', 'M19', '(S6, C0, G0, P1)')
_try_import('.M20_S6_C0_G0_P2.model', 'M20', '(S6, C0, G0, P2)')
_try_import('.M25_S6_CT_G0_P1.model', 'M25', '(S6, CT, G0, P1)')
_try_import('.M26_S6_CT_G0_P2.model', 'M26', '(S6, CT, G0, P2)')
_try_import('.M27_S6_CT_GC_P1.model', 'M27', '(S6, CT, GC, P1)')
_try_import('.M28_S6_CT_GC_P2.model', 'M28', '(S6, CT, GC, P2)')
