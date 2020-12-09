class Parameters:
    # Configurations
    N_CHANNELS = 10
    # Transceiver technology
    FIXED_RATE_TRANS = "fixed_rate"
    FLEX_RATE_TRANS = "flex_rate"
    SHANNON_TRANS = "shannon"


class SigConstants:
    # Noise bandwidth
    Bn = 12.5e9
    # Ref. symbol rate
    Rs = 32e9
    # BER threshold for FEC
    BERt = 2.5
