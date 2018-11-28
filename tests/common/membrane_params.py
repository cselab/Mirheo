
def set_lina(lscale, prms):
    p              = 0.000906667 * lscale
    prms.x0        = 0.457    
    prms.ka        = 4900.0    
    prms.kd        = 5000
    prms.kv        = 7500.0
    prms.gammaC    = 52.0 * lscale
    prms.gammaT    = 0.0
    prms.kbT       = 0.0444 * lscale**2
    prms.mpow      = 2.0
    prms.totArea   = 62.2242 * lscale**2
    prms.totVolume = 26.6649 * lscale**3
    prms.ks        = prms.kbT / p
    prms.rnd       = False

def set_lina_bending(lscale, prms):
    prms.kb        = 44.4444 * lscale**2
    prms.theta     = 6.97    
