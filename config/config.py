import os


class config_container(object):

    def __iter__(self):
        for _, value in self.__dict__.iteritems():
            if value:
                yield value

    def __getattr__(self, item):
        return None

    def _to_dict(self):
        d = {}
        for attr, value in self.__dict__.iteritems():
            if isinstance(value, config_container):
                d[attr] = value._to_dict()
            else:
                d[attr] = value
        return d

    def __repr__(self):
        import json
        return json.dumps(self._to_dict(), indent=2)


def base_model_config():
    config = config_container()
    config.input_size = 28
    config.lstm_units = 300
    config.num_hidden_layers = 4
    config.max_time_steps = 20
    config.init_scale = 1.0
    config.max_grad_norm = 5
    config.learning_rate = 0.002
    config.keep_prob = 0.8
    config.batch_size = 20
    config.load_mode = "best"
    config.patience = 3
    return config


def config():
    config = config_container()
    config.base_model_config = base_model_config() 
    return config


if __name__ == '__main__':
    config = config()
    print(config)