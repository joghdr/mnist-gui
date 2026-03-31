from core.Config import Config


class TestConfig():

    REQUIRED_ATTRS = [
        "activation_hidden",
        "batch_size",
        "cost",
        "digits",
        "epochs",
        "input_dimension",
        "learning_rate",
        "output_dimension",
        "print_steps",
        "ratio",
        "runid",
        "sample",
        "sample_test",
    ]

    def test_eq_identical_instances(self):
        config = Config()
        config_new = Config()
        assert config == config_new

    def test_eq_different_instances(self):
        config = Config()
        for attr in self.REQUIRED_ATTRS:
          config_new = Config()
          value = getattr(config, attr)

          if isinstance(value, str):
              setattr(config_new, attr, value + "_suffix")
          elif isinstance(value, (int, float)):
              setattr(config_new, attr, value + 1)
          elif isinstance(value, bool):
              setattr(config_new, attr, not value)
          elif isinstance(value, list):
              setattr(config_new, attr, value + [1])
          else:
              raise TypeError(f'Test for {attr} of type {type(value)} not implemented.')

          assert config != config_new

    def test_attributes_required(self):
        config = Config()
        for attr in self.REQUIRED_ATTRS:
          assert hasattr(config, attr)

    def test_critical_values(self):
        config = Config()
        assert 0 < config.ratio < 1
        assert config.batch_size > 0
        assert config.learning_rate > 0
        for number in config.digits:
            assert number in range(10)
