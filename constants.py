scale1 = 1.25
inv_scal1e = 0.8

scale2 = 1.5625
inv_scale2 = 0.64

scale3 = 1.953125
inv_scale3 = 0.512

scale4 = 2.44140625
inv_scale4 = 0.4096

scale5 = 3.0517578125
inv_scale5 = 0.32768

scale6 = 3.814697265625
inv_scale6 = 0.262144

w = 480
scale = 1.25
while True:
    w1 = w/scale
    print(scale)
    print(1/scale)
    scale = scale * scale1
    if w1 < 24:
        break

