"""
spm_eval/optimiser.py
---------------------
Parameter optimisation for SPM reconstruction models.

ParameterOptimiser: search for the best parameter values
                    for a given model on a given collection.

Uses random search by default. Bayesian search available if optuna
is installed: pip install optuna

Usage
-----
    from spm_eval.optimiser import ParameterOptimiser
    from spm_data.collection import SPMCollection
    from spm_eval.evaluator  import Evaluator
    from spm_reconstruct     import get_model

    col   = SPMCollection(Ny=64, Nx=64)
    ev    = Evaluator()
    model_cls = type(get_model("(S6, CT, GC, P1)"))

    opt = ParameterOptimiser(model_cls, col, ev)
    result = opt.optimise(n_trials=50, metric='rmse', tier=1)
    print(result['best_params'])
    print(result['best_value'])
"""

import numpy as np


class ParameterOptimiser:
    """
    Optimise tunable parameters for a given model class.

    Parameters
    ----------
    model_cls  : class — the algorithm class (e.g. M27)
    collection : SPMCollection
    evaluator  : Evaluator
    """

    def __init__(self, model_cls, collection, evaluator):
        self.model_cls  = model_cls
        self.collection = collection
        self.evaluator  = evaluator

    def optimise(self,
                 n_trials=50,
                 metric='rmse',
                 tier=1,
                 fixed_params=None,
                 method='random',
                 verbose=True) -> dict:
        """
        Search for best parameters.

        Parameters
        ----------
        n_trials     : int   — number of parameter configurations to try
        metric       : str   — metric to minimise ('rmse', 'nll')
                               or maximise ('ssim', 'coverage95', 'classifier_auc')
        tier         : int   — collection tier to evaluate on
        fixed_params : dict  — parameters held constant during search
        method       : str   — 'random' or 'bayesian' (requires optuna)
        verbose      : bool  — print progress

        Returns
        -------
        dict with keys:
            best_params : dict  — best parameter values found
            best_value  : float — best metric value achieved
            metric      : str   — which metric was optimised
            all_trials  : list  — [(params, value)] for all trials
        """
        MAXIMISE = {'ssim', 'coverage95', 'coverage68', 'classifier_auc'}
        maximise = metric in MAXIMISE

        tunable = [p for p in self.model_cls.PARAM_SPACE
                   if p.get('category') == 'tuning'
                   and p.get('bounds') is not None]

        if not tunable:
            raise ValueError(
                f"{self.model_cls.MODEL_ID} has no tunable parameters "
                "with bounds defined in PARAM_SPACE.")

        fixed = fixed_params or {}

        if method == 'bayesian':
            return self._optimise_bayesian(
                tunable, fixed, n_trials, metric, tier, maximise, verbose)
        else:
            return self._optimise_random(
                tunable, fixed, n_trials, metric, tier, maximise, verbose)

    def _sample_random(self, tunable, rng):
        """Sample one parameter configuration uniformly at random."""
        params = {}
        for p in tunable:
            lo, hi = p['bounds']
            if p.get('dtype') == int:
                params[p['name']] = int(rng.integers(lo, hi+1))
            else:
                params[p['name']] = float(rng.uniform(lo, hi))
        return params

    def _evaluate_params(self, params, tier, metric):
        """Evaluate one parameter configuration. Returns scalar."""
        try:
            model  = self.model_cls(params=params)
            report = self.evaluator.evaluate_collection(model,
                                                        self.collection,
                                                        tier=tier)
            return report.mean(metric)
        except Exception:
            return float('nan')

    def _optimise_random(self, tunable, fixed, n_trials,
                         metric, tier, maximise, verbose):
        rng        = np.random.default_rng(0)
        best_val   = -np.inf if maximise else np.inf
        best_params = None
        all_trials  = []

        if verbose:
            print(f"Random search: {n_trials} trials, "
                  f"metric={metric} ({'max' if maximise else 'min'})")

        for trial in range(n_trials):
            params = self._sample_random(tunable, rng)
            params.update(fixed)

            val = self._evaluate_params(params, tier, metric)

            is_better = (val > best_val) if maximise else (val < best_val)
            if is_better and not np.isnan(val):
                best_val    = val
                best_params = params.copy()

            all_trials.append((params.copy(), val))

            if verbose and (trial+1) % 10 == 0:
                print(f"  Trial {trial+1}/{n_trials}  "
                      f"best {metric}={best_val:.5f}")

        return {
            'best_params': best_params,
            'best_value':  best_val,
            'metric':      metric,
            'method':      'random',
            'all_trials':  all_trials,
        }

    def _optimise_bayesian(self, tunable, fixed, n_trials,
                           metric, tier, maximise, verbose):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("optuna not installed — falling back to random search.")
            print("Install with: pip install optuna")
            return self._optimise_random(
                tunable, fixed, n_trials, metric, tier, maximise, verbose)

        def objective(trial):
            params = {}
            for p in tunable:
                lo, hi = p['bounds']
                if p.get('dtype') == int:
                    params[p['name']] = trial.suggest_int(p['name'], lo, hi)
                else:
                    params[p['name']] = trial.suggest_float(p['name'], lo, hi)
            params.update(fixed)
            val = self._evaluate_params(params, tier, metric)
            return val if not np.isnan(val) else (
                -np.inf if maximise else np.inf)

        direction = 'maximize' if maximise else 'minimize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials,
                       show_progress_bar=verbose)

        best = study.best_trial
        return {
            'best_params': dict(best.params, **fixed),
            'best_value':  best.value,
            'metric':      metric,
            'method':      'bayesian',
            'all_trials':  [(t.params, t.value) for t in study.trials],
        }
