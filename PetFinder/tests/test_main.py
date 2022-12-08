import pytest
from tests.test_preprocessing import PPF, train_ds


def test_model_setup(PPF, train_ds):
    print(train_ds)

# def test_model_binary(PPF, layers, train_ds):
#     model = PPF.model_setup(*layers)
#     model = PPF.model_compilers(model)
#     model = PPF.model_fit(model, train_ds)
#     assert 0

# def test_model_sparse(PPF, layers, train_ds):
#     model = PPF.model_setup_sparse(*layers)
#     model = PPF.model_compilers_sparse(model)
#     model = PPF.model_fit(model, train_ds)
#     assert 0
