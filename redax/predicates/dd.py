
try:
    from dd.cudd import BDD
except ImportError:
    import warnings
    warnings.warn("dd not using CUDD. Slow runtimes expected", RuntimeWarning)
    from dd.autoref import BDD
