from .testing_parameters import TestingParameters

TEST_EXPERIMENT = [
    TestingParameters(model_name='TestModel1', epochs=5),
    TestingParameters(model_name='TestModel2', epochs=10),
    TestingParameters(model_name='TestModel3', epochs=20),
]

TWINKLE_TWINKLE_EXPERIMENT = [
    TestingParameters(model_name='Twinkle5', epochs=5),
    TestingParameters(model_name='Twinkle10', epochs=10),
    TestingParameters(model_name='Twinkle20', epochs=20),
    TestingParameters(model_name='Twinkle35', epochs=35),
    TestingParameters(model_name='Twinkle50', epochs=50),
]

EPOCH_EXPERIMENT = [
    TestingParameters(epochs=10),
    TestingParameters(epochs=15),
    TestingParameters(epochs=20),
    TestingParameters(epochs=25),
    TestingParameters(epochs=30),
]
