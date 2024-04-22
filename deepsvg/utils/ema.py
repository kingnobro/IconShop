class Ema():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and not ('argument_fcn' in name or 'argument_decoder' in name):
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and not ('argument_fcn' in name or 'argument_decoder' in name):
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and not ('argument_fcn' in name or 'argument_decoder' in name):
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and not ('argument_fcn' in name or 'argument_decoder' in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}