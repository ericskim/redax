
def test_import_cudd():

    try:
        from dd.cudd import BDD
    except ImportError:
        import warnings
        warnings.warn("dd not using CUDD. Slow runtimes expected", RuntimeWarning)
        from dd.autoref import BDD


def test_import_dd():
    import dd