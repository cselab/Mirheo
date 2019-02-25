
def lina_parameters(lscale=1.0, fluctuations=False):
    p = 0.000906667 * lscale
    kBT = 0.0444 * lscale**2
    prms = {
        "x0"     : 0.457,
        "ka_tot" : 4900.0,
        "kv_tot" : 7500.0,
        "ka"     : 5000,
        "ks"     : kBT / p,
        "mpow"   : 2.0,
        "gammaC" : 52.0 * lscale,
        "gammaT" : 0.0,
        "kBT"    : 0.0,
        "tot_area"   : 62.2242 * lscale**2,
        "tot_volume" : 26.6649 * lscale**3,
        "kb"     : 44.4444 * lscale**2,
        "theta"  : 6.97
    }
    if fluctuations:
        prms["kBT"] = kBT
    
    return prms
