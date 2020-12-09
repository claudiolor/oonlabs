import numpy as np


class SignalUtils:
    @staticmethod
    def snr(signal_power, noise_power):
        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def noise(signal_power, length):
        return 1e-9 * signal_power * length

    @staticmethod
    def latency(length):
        LIGHTSPEED2_3 = 199861638.67
        return length / LIGHTSPEED2_3
