import numpy as np

def LQ(y):
    x = y.copy()
    rate = {0:0, 1:12, 2:24, 3:48, 4:196}
    for i in range(len(x)):
        if x[i]<rate[1]:
            x[i] = rate[0]
        elif x[i]<rate[2]:
            x[i] = rate[1]
        elif x[i]<rate[3]:
            x[i] = rate[2]
        elif x[i]<rate[4]:
            x[i] = rate[3]
        else:
            x[i] = rate[4]
    return x

def UQ(y):
    x = y.copy()
    rate = {0:0, 1:12, 2:24, 3:48, 4:196}
    for i in range(len(x)):
        if x[i]>rate[3]:
            x[i] = rate[4]
        elif x[i]>rate[2]:
            x[i] = rate[3]
        elif x[i]>rate[1]:
            x[i] = rate[2]
        elif x[i]>rate[0]:
            x[i] = rate[1]
        else:
            x[i] = rate[0]
    return x

def A_to_R(attn,r):
    """
    Map attention scores to resolution levels.
    
    Args:
        attn (np.ndarray): 2D attention map
        r (float): target rate budget

    Returns:
        np.ndarray: resolution mask (same shape as attn)
    """
    rate = {0:0, 1:12, 2:24, 3:48, 4:196}
    attn_flatten = attn.reshape([2400])
    sum_attn = attn_flatten.sum()
    attn_f = attn_flatten*(r/sum_attn)
    mask = np.zeros(attn_f.shape)
    attn_f_perm = np.argsort(-attn_f)
    if r <= rate[1]*2400:
        for ind in range(1,int(r/rate[1])):
            mask[attn_f_perm[ind]] = 1
    else:
        # Replace with LQ but once the LQ is 0, we replace it with 1
        LQ_attn = LQ(attn_f)
        UQ_attn = UQ(attn_f)
        attn_f[LQ_attn==0] = -1000
        attn_f[LQ_attn==196] = -10000
        LQ_attn[LQ_attn==0] = rate[1] # We will overshoot r because of replacing 0 with 1.
        sum_LQ_rate = sum(LQ_attn)
        while sum_LQ_rate<r: # We might overshoot in the last loop
            diff = UQ_attn - attn_f
            diff_perm = np.argsort(diff)
            LQ_attn[diff_perm[0]] = UQ_attn[diff_perm[0]]
            attn_f[diff_perm[0]] = LQ_attn[diff_perm[0]] + 0.01
            if UQ_attn[diff_perm[0]] == 196:
                attn_f[diff_perm[0]] = -10000
            sum_LQ_rate = sum(LQ_attn)
            UQ_attn = UQ(attn_f)
            
    if r > rate[1]*2400:
        LQ_attn[LQ_attn==rate[1]] = 1
        LQ_attn[LQ_attn==rate[2]] = 2
        LQ_attn[LQ_attn==rate[3]] = 3
        LQ_attn[LQ_attn==rate[4]] = 4
    else:
        LQ_attn = mask
    mask_attn = LQ_attn.reshape([40,60])
    
    
    return mask_attn
