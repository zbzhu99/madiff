class Encoder(object):
    def __call__(self, data):
        raise NotImplementedError


class IdentityEncoder(Encoder):
    def __call__(self, data):
        return data


class SMAC5m6mEncoder(Encoder):
    def __call__(self, data):
        data[..., 1:, :, :5] = data[..., 0:1, :, :5]
        return data


class SMAC3mEncoder(Encoder):
    def __call__(self, data):
        data[..., 1:, :, :3] = data[..., 0:1, :, :3]
        return data
