import random
import numpy as np


def calc_poly_v6(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = pts[30]
    y10 = h*0.97
    y6 =  y10
    y8 =  y10
    
    p13 = pts[13] - rand(0.3, 0.4)*(pts[13] - p0)
    p12 = pts[12] - rand(0.1, 0.15)*(pts[12] - p0)
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - p0)
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - p0)
    p05[0] = pts[5][0]
    p04 = pts[4] - rand(0.1, 0.15)*(pts[4] - p0)
    p03 = pts[3] - rand(0.3, 0.4)*(pts[3] - p0)
    
    pts = [p0, p0, p13, p12, p11, p10, p08, p06, p05, p04, p03, p03]
    return np.round(np.array(pts)).astype(np.int32)

def calc_poly_v7(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = pts[29]
    y10 = h*0.97
    y6 =  y10
    y8 =  y10
    
    p14 = pts[14] - rand(0.05, 0.10)*(pts[14] - p0)
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - p0)
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - p0)
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - p0)
    p05[0] = pts[5][0]
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - p0)
    p02 = pts[2] - rand(0.05, 0.10)*(pts[2] - p0)
    
    pts = [p0, p0, p14, p12, p11, p10, p08, p06, p05, p04, p02, p02]
    return np.round(np.array(pts)).astype(np.int32)

def calc_poly_v8(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = pts[27]
    y10 = h*0.97
    y6 =  y10
    y8 =  y10
    
    p14 = pts[15] - rand(0.05, 0.10)*(pts[14] - p0)
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - p0)
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - p0)
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - p0)
    p05[0] = pts[5][0]
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - p0)
    p02 = pts[1] - rand(0.05, 0.10)*(pts[2] - p0)
    
    pts = [p0, p0, p14, p12, p11, p10, p08, p06, p05, p04, p02, p02]
    return np.round(np.array(pts)).astype(np.int32)

def calc_poly_v9(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = pts[27]
    y10 = h*0.99
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = pts[5][0]
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.10)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    
    pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)


# 김경수 아나운서 데모용
def calc_poly_v10(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = pts[27]
    y10 = h*0.99
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = (pts[5][0] + pts[6][0])/2
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.10)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    
    pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)


# CNA 아나운서 데모용
def calc_poly_v11(pts, h, randomness=False):
    
    def rand(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    p0 = pts[27]
    y10 = h*0.99
    y6 =  y10
    y8 =  y10
    
    #print('4')
    #p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p15 = pts[15] - rand(0.08, 0.12)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = (pts[5][0] + pts[6][0])/2
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.10)*(pts[3] - pts[30])
    #p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    p01 = pts[1] - rand(0.08, 0.12)*(pts[1] - p0)
    
    pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)

# 이민영 강사
def calc_poly_v21(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = pts[27]
    y10 = h*0.99
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] + rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])
    
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = (pts[5][0] + pts[6][0])/2
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.05)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.05)*(pts[1] - p0)
    
    p05[0] = p03[0]
    p04[0] = p03[0]
    
    p11[0] = p13[0]
    p12[0] = p13[0]
    
    pts = [p0,  # 미간
           p15, p13, p12, p11, # 우리가 보기에 오른쪽. p15가 위, p11이 아래 
           p10, p08, p06, # 맨 아래 
           p05, p04, p03, p01] # 우리가 보기에 왼쪽, p01 이 위, p05가 아래
    return np.round(np.array(pts)).astype(np.int32)

# 이민영 강사, 측면 머리카락 안 가리게 
def calc_poly_v22(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = pts[27]
    y10 = h*0.99
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.10)*(pts[13] - pts[30])
    #p12 = pts[12] + rand(0.01, 0.05)*(pts[12] - pts[30])
    p12 = pts[12] + rand(0.10, 0.10)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.12, 0.12)*(pts[11] - pts[30])
    
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = (pts[5][0] + pts[6][0])/2
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.05)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.05)*(pts[1] - p0)
    
    p05[0] = p03[0]
    p04[0] = p03[0]
    
    #p11[0] = p13[0]
    #p12[0] = p13[0]
    
    pts = [p0,  # 미간
           p15, p13, p12, p11, # 우리가 보기에 오른쪽. p15가 위, p11이 아래 
           p10, p08, p06, # 맨 아래 
           p05, p04, p03, p01] # 우리가 보기에 왼쪽, p01 이 위, p05가 아래
    return np.round(np.array(pts)).astype(np.int32)

# 이민영 강사, 측면 위 22번 마스크에서 얼굴을 좀 더 가리게 p13번을 밖으로 더 빼냄.
def calc_poly_v23(pts, h, randomness=True):
    
    def rand_(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e)
    
    p0 = pts[27]
    y10 = h*0.99
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] + rand(0.01, 0.10)*(pts[13] - pts[30])
    #p12 = pts[12] + rand(0.01, 0.05)*(pts[12] - pts[30])
    p12 = pts[12] + rand(0.10, 0.10)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.12, 0.12)*(pts[11] - pts[30])
    
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = (pts[5][0] + pts[6][0])/2
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.05)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.05)*(pts[1] - p0)
    
    p05[0] = p03[0]
    p04[0] = p03[0]
    
    #p11[0] = p13[0]
    #p12[0] = p13[0]
    
    pts = [p0,  # 미간
           p15, p13, p12, p11, # 우리가 보기에 오른쪽. p15가 위, p11이 아래 
           p10, p08, p06, # 맨 아래 
           p05, p04, p03, p01] # 우리가 보기에 왼쪽, p01 이 위, p05가 아래
    return np.round(np.array(pts)).astype(np.int32)

# 이민영 강사, 측면 위 23번 마스크에서 얼굴을 좀 더 가리게 p15번을 밖으로 더 빼냄.
def calc_poly_v24(pts, h, randomness=True):
    
    def rand_(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e)
    
    p0 = pts[27]
    y10 = h*0.99
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] + rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] + rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] + rand(0.10, 0.10)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.12, 0.12)*(pts[11] - pts[30])
    
    p11[0] = pts[11][0]
    p10 = [pts[9][0], y10]
    
    p08 = [pts[8][0], y8]
    
    p06 = [pts[7][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = (pts[5][0] + pts[6][0])/2
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.05)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.05)*(pts[1] - p0)
    
    p05[0] = p03[0]
    p04[0] = p03[0]
    
    #p11[0] = p13[0]
    #p12[0] = p13[0]
    
    pts = [p0,  # 미간
           p15, p13, p12, p11, # 우리가 보기에 오른쪽. p15가 위, p11이 아래 
           p10, p08, p06, # 맨 아래 
           p05, p04, p03, p01] # 우리가 보기에 왼쪽, p01 이 위, p05가 아래
    return np.round(np.array(pts)).astype(np.int32)

# 박은보, 옆모습, 아주 많이 가림
def calc_poly_pwb_side_v39(pts, h, randomness=True):
    
    def rand(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
    
    p0 = pts[27]
    y10 = h*0.85
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] + rand(0.01, 0.05)*(pts[15] - p0)#- rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] + rand(0.01, 0.05)*(pts[13] - pts[30])#- rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] + rand(0.05, 0.10)*(pts[12] - pts[30])#- rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])#+ rand(0.18, 0.22)*(pts[11] - pts[30])
    p11[0] = pts[11][0]
    p10 = [pts[10][0], y10]
    p08 = [pts[9][0], y8]
    p06 = [pts[8][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = pts[5][0]
    p04 = pts[4] + rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] + rand(0.01, 0.10)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    
    #pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    pts = [p0, p15, p13, p12, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)


# 박은보, 옆모습, 많이 가리지 않음, 예전 v9와 유사
def calc_poly_pwb_side_v39_1(pts, h, randomness=True):
    
    def rand(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    p0 = pts[27]
    y10 = h*0.85
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])
    p11[0] = pts[11][0]
    p10 = [pts[10][0], y10]
    p08 = [pts[9][0], y8]
    p06 = [pts[8][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = pts[5][0]
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.10)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    
    #pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    pts = [p0,  
           p15, p13, p12, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)

def calc_poly_pwb_front_v39_0(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = [pts[27][0], pts[27][1] + h*0.09]
    #p0 = pts[27]
    y10 = h*0.87
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.08)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.08)*(pts[13] - pts[30])
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - pts[30])
    p12[0] = p13[0] + h*0.01
    
    
    p10 = [pts[9][0], y10]
    p08 = [pts[8][0], y8]
    p06 = [pts[7][0], y6]
    
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    p03 = pts[3] - rand(0.01, 0.10)*(pts[3] - pts[30])
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p04[0] = p03[0] - h*0.01
    
    
    pts = [p0,  # 미간
           p15, p13, p12,
           # p11, # 우리가 보기에 오른쪽. p15가 위, p11이 아래 
           p10, p08, p06, # 맨 아래 
           #p05,
           p04, p03, p01] # 우리가 보기에 왼쪽, p01 이 위, p05가 아래
        
        
    return np.round(np.array(pts)).astype(np.int32)


# 박은보, 옆모습, 아주 많이 가림
def calc_poly_pwb_side_v39(pts, h, randomness=True):
    
    def rand(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
    
    p0 = pts[27]
    y10 = h*0.85
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] + rand(0.01, 0.05)*(pts[15] - p0)#- rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] + rand(0.01, 0.05)*(pts[13] - pts[30])#- rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] + rand(0.05, 0.10)*(pts[12] - pts[30])#- rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])#+ rand(0.18, 0.22)*(pts[11] - pts[30])
    p11[0] = pts[11][0]
    p10 = [pts[10][0], y10]
    p08 = [pts[9][0], y8]
    p06 = [pts[8][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = pts[5][0]
    p04 = pts[4] + rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] + rand(0.01, 0.10)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    
    #pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    pts = [p0, p15, p13, p12, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)


# 박은보, 옆모습, 많이 가리지 않음, 예전 v9와 유사
def calc_poly_pwb_side_v39_1(pts, h, randomness=True):
    
    def rand(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    p0 = pts[27]
    y10 = h*0.85
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.01, 0.10)*(pts[15] - p0)
    p13 = pts[13] - rand(0.01, 0.10)*(pts[13] - pts[30])
    p12 = pts[12] - rand(0.01, 0.05)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.18, 0.22)*(pts[11] - pts[30])
    p11[0] = pts[11][0]
    p10 = [pts[10][0], y10]
    p08 = [pts[9][0], y8]
    p06 = [pts[8][0], y6]
    p05 = pts[5] + rand(0.18, 0.22)*(pts[5] - pts[30])
    p05[0] = pts[5][0]
    p04 = pts[4] - rand(0.01, 0.05)*(pts[4] - pts[30])
    p03 = pts[3] - rand(0.01, 0.10)*(pts[3] - pts[30])
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    
    #pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    pts = [p0,  
           p15, p13, p12, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)


# 박은보, 측면, 아주 많이 가림
# calc_poly_pwb_side_v39 보다 더 많이 가림
# y10 : 99 -> 0.80 로 함
def calc_poly_pwb_side_v39_11(pts, h, randomness=True):
    
    def rand(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
    
    p0 = pts[27]
    y10 = h*0.80
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] + rand(0.04, 0.08)*(pts[15] - p0)
    p13 = pts[13] + rand(0.04, 0.08)*(pts[13] - pts[30])
    p12 = pts[12] + rand(0.04, 0.08)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.04, 0.08)*(pts[11] - pts[30])
    p11[0] = pts[11][0] + rand(0.18, 0.22)*(pts[11][0] - pts[30][0])
    
    p10 = pts[10] + rand(0.12, 0.18)*(pts[10] - pts[30])
    p08 = pts[9]  + rand(0.12, 0.18)*(pts[9] - pts[30])
    p06 = pts[8]  + rand(0.12, 0.18)*(pts[8] - pts[30])
    p05 = pts[5]  + rand(0.12, 0.18)*(pts[5] - pts[30])
    p05[0] = pts[5][0]  + rand(0.18, 0.22)*(pts[5][0] - pts[30][0])
    
    p04 = pts[4]  + rand(0.12, 0.18)*(pts[4] - pts[30])
    p03 = pts[3]  + rand(0.12, 0.18)*(pts[3] - pts[30])
    p01 = pts[1]  + rand(0.12, 0.18)*(pts[1] - p0)
    
    #p11[0] = pts[11][0]
    p10[1] = y10
    p08[1] = y10
    p06[1] = y10 
    #p05[0] = pts[5][0]
    
    pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    #pts = [p0, p15, p13, p12, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)


def calc_poly_pwb_side_v39_12(pts, h, randomness=True):
    
    def rand(a, b):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
    
    p0 = pts[27]
    y10 = h*0.80
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] + rand(0.20, 0.28)*(pts[15] - p0)
    p13 = pts[13] + rand(0.20, 0.28)*(pts[13] - pts[30])
    p12 = pts[12] + rand(0.20, 0.28)*(pts[12] - pts[30])
    p11 = pts[11] + rand(0.20, 0.28)*(pts[11] - pts[30])
    p11[0] = pts[11][0] + rand(0.18, 0.22)*(pts[11][0] - pts[30][0])
    
    p10 = pts[10] + rand(0.12, 0.18)*(pts[10] - pts[30])
    p08 = pts[9]  + rand(0.12, 0.18)*(pts[9] - pts[30])
    p06 = pts[8]  + rand(0.12, 0.18)*(pts[8] - pts[30])
    p05 = pts[5]  + rand(0.12, 0.18)*(pts[5] - pts[30])
    p05[0] = pts[5][0]  + rand(0.18, 0.22)*(pts[5][0] - pts[30][0])
    
    p04 = pts[4]  + rand(0.12, 0.18)*(pts[4] - pts[30])
    p03 = pts[3]  + rand(0.12, 0.18)*(pts[3] - pts[30])
    p01 = pts[1]  + rand(0.12, 0.18)*(pts[1] - p0)
    
    #p11[0] = pts[11][0]
    p10[1] = y10
    p08[1] = y10
    p06[1] = y10 
    #p05[0] = pts[5][0]
    
    pts = [p0, p15, p13, p12, p11, p10, p08, p06, p05, p04, p03, p01]
    #pts = [p0, p15, p13, p12, p10, p08, p06, p05, p04, p03, p01]
    return np.round(np.array(pts)).astype(np.int32)


def calc_poly_pwb_front_v39_1(pts, h, randomness=True):
    
    def rand_(a, b, randomness=True):
        if randomness:
            r = random.random()
            return a + (b-a)*r
        else:
            return (a+b)/2
        
    def rand(s, e):
        return rand_(s, e, randomness)
    
    p0 = [pts[27][0], pts[27][1] + h*0.09]
    #p0 = pts[27]
    y10 = h*0.9
    y6 =  y10
    y8 =  y10
    
    p15 = pts[15] - rand(0.02, 0.06)*(pts[15] - p0)
    p12 = [p15[0], p15[1] + (y10 - p15[1])*0.5] 
    
    
    p10 = [h*0.65, y10]
    p08 = [h*0.5, y8]
    p06 = [h*0.35, y6]
    
    p01 = pts[1] - rand(0.01, 0.10)*(pts[1] - p0)
    p04 = [p01[0], p01[1] + (y10-p01[1])*0.5]
    
    
    pts = [p0,  # 미간
           p15, p12, # 우리가 보기에 오른쪽, p15가 위
           p10, p08, p06, # 맨 아래 
           p04, p01] # 우리가 보기에 왼쪽, p01 이 위, p04가 아래
        
        
    return np.round(np.array(pts)).astype(np.int32)

calc_poly = {
    6:calc_poly_v6,
    7:calc_poly_v7,
    8:calc_poly_v8,
    9:calc_poly_v9,
    10:calc_poly_v10,
    11:calc_poly_v11,
    21:calc_poly_v21,
    22:calc_poly_v22,
    23:calc_poly_v23,
    24:calc_poly_v24,
    'pwb_side_v39': calc_poly_pwb_side_v39,
    'pwb_side_v39_1': calc_poly_pwb_side_v39_1,
    'pwb_side_v39_11': calc_poly_pwb_side_v39_11,
    'pwb_side_v39_12': calc_poly_pwb_side_v39_12,
    'pwb_front_v39_0': calc_poly_pwb_front_v39_0,
    'pwb_front_v39_1': calc_poly_pwb_front_v39_1
}
