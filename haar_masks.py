#coordinates: x,y,W,H... ups
#trebuie pad, ca altfel numba se supara
import numpy as np

feature1 = [ # Detectare de sprancene / ochi fata de nas. 2 dreptunghiuri. alb si negru. 24x3 ambele. suma: 0. de ce? detectam contrast. daca e 0 => nu e diferenta de pixeli acolo
    (0, 6, 24, 3, -1), (0, 9, 24, 3, +1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)
]

feature2 = [ # Nas. 2 dreptunghiuri 8x6.
    #UPDATE: 24 - 24
    (9, 10, 4, 6, +1), (13, 10, 4, 6, -1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)
]

feature3 = [ # Ochi. 3 dreptunghiuri. 2 de 4x8 si un 12x8. Ca sa pastram balanta, facem weight uri diferite :
    # (32 * 1.5) - (96 * 1) + (32 * 1.5) = 48 - 96 + 48 = 0
    # UPDATED: 48*(-1) + 32 * (+3) + 48 * (-1) = 0
    (5, 8, 6, 8, -1.0), (11, 8, 4, 8, +3.0), (15, 8, 6, 8, -1.0), (0, 0, 0, 0, 0)
]
feature4 = [#3 dreptunghiuri, negru si mare in mijloc, albe mici pe margini: ochii, dar acum de sus in jos => buze
    (4, 17, 16, 2, +3), (4, 19, 16, 3, -4), (4, 22, 16, 2, +3), (0, 0, 0, 0, 0)
]
feature5 = [ # jucam sah pe moaca '-' -> jawline
    (16, 16, 2, 2, +1),
    (18, 16, 2, 2, -1),
    (16, 18, 2, 2, -1),
    (18, 18, 2, 2, +1)
]
actual_feature_count = [2,2,3,3,4]#trebuie sa aiba aceeasi lungime sa nu se sperie np sau njit, dar, de fapt, sunt doar count feature uri per masca
features = [feature1, feature2, feature3, feature4, feature5]
haar_features = np.array(features, dtype=np.float64)
haar_counts = np.array(actual_feature_count, dtype=np.int64)


