import numpy as np
import scipy.special as scys
from include.parameters import SigConstants as sc
from include.parameters import Parameters as const


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

    @staticmethod
    def ase_noise(G, NF, Namp):
        return Namp * (sc.h * sc.CBcenter * sc.Bn * NF * (G-1))

    @staticmethod
    def eta_nli(Rs, df):
        Leff = 1 / (2 * sc.alpha)
        return (16 / (27 * np.pi)) * \
               np.log(((np.pi ** 2) / 2) *
                      ((sc.beta2 * (Rs ** 2))/sc.alpha) *
                      (const.N_CHANNELS ** (2*(Rs/df)))) * \
               (sc.alpha/sc.beta2) * (((sc.gamma ** 2) * (Leff ** 2)) / (Rs ** 3))

    @staticmethod
    def nli_noise(eta_nli, ch_power, Nspan):
        return eta_nli * (ch_power ** 3) * Nspan * sc.Bn


class TransceiverCharacterization:
    @staticmethod
    def fixed_rate100(BERt, Rs, Bn):
        return 2 * (scys.erfcinv(2 * BERt) ** 2) * (Rs / Bn)

    @staticmethod
    def flex_rate0(BERt, Rs, Bn):
        return 2 * (scys.erfcinv(2 * BERt) ** 2) * (Rs / Bn)

    @staticmethod
    def flex_rate100(BERt, Rs, Bn):
        return (14 / 3) * (scys.erfcinv((3 / 2) * BERt) ** 2) * (Rs / Bn)

    @staticmethod
    def flex_rate200(BERt, Rs, Bn):
        return 10 * (scys.erfcinv((8 / 3) * BERt) ** 2) * (Rs / Bn)

    @staticmethod
    def shannon(Rs, Bn, snr):
        return 2 * Rs * np.log2(1 + (snr * (Bn / Rs)))
