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
    config.input_size = 17
    config.lstm_units = 128
    config.num_hidden_layers = 3
    config.max_time_steps = 20
    config.init_scale = 1.0
    config.max_grad_norm = 5
    config.learning_rate = 0.01
    config.keep_prob = 0.7
    config.batch_size = 25
    config.load_mode = "fresh"
    config.patience = 9
    config.output_size = 1
    config.forced_size = 5
    return config

def dkf_model_config():
    config = base_model_config()

    config.batch_size = 20
    config.max_time_steps = 20

    config.nsamples_e1 = 20
    config.nsamples_e3 = 20
    config.lsm_time = 10

    config.latent_state_size = 10

    config.input_size = 17
    config.output_size = 1

    config.learning_rate = 0.0000003
    config.keep_prob = 0.7

    config.num_hidden_units = 200    
    config.num_hidden_layers = 3

    config.max_grad_norm = 1000

    config.logfolder = "log/"
    
    config.load_mode = "train"    
    return config

def config():
    config = config_container()
    config.base_model_config = base_model_config() 
    config.dkf_model_config = dkf_model_config() 
    return config


if __name__ == '__main__':
    config = config()
    print(config)
