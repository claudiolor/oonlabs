class Parameters:
    # Configurations
    N_CHANNELS = 10
    # Transceiver strategy
    FIXED_RATE_TRANS = "fixed_rate"
    FLEX_RATE_TRANS = "flex_rate"
    SHANNON_TRANS = "shannon"


class ConnectionStatus:
    DEPLOYED = 0
    PENDING = 1
    BLOCKING_EVENT = 2
    LOW_SNR = 3


class SigConstants:
    # Noise bandwidth
    Bn = 12.5e9
    # Ref. symbol rate
    Rs = 32e9
    # BER threshold for FEC
    BERt = 1e-3
    # Number of km for each amplifier
    KmPerA = 80e3
    # Amplifier default gain 16dB in linear units
    AdefGain = 39.81
    # Noise figure 3dB in linear units
    NF = 1.995
    # Plank constant m^2 * kg / s
    h = 6.626e-34
    # C band center
    CBcenter = 193.414e12
    # Frequency spacing
    df = 50e9
    # Speed of the light in the glass m/s
    lightspeed2_3 = 199861638.67

    ##
    # Characteristic of the fiber
    ##
    # Fiber alpha value in W/m
    alpha = 2.3e-5
    # Beta2 of the fiber (m*Hz^2)^-1
    beta2 = 2.13e-26
    # Gamma value of the fiber in (W*m)^-1
    gamma = 1.27e-3
