def calculate_frequencies(s, d):
    c = 344
    fu_term1 = (0.58 * c) / d
    fu_values = []
    fl_values = []
    for s_val in s:
        fu_term2 = (0.45 * c) / s_val
        fu = min(fu_term1, fu_term2)
        fl = (0.01 * c) / s_val
        fu_values.append(fu)
        fl_values.append(fl)
    max_fu = max(fu_values)
    min_fl = min(fl_values)
    return max_fu, min_fl

s=[0.080,0.015]
d=0.02
fu,fl = calculate_frequencies(s, d)
print(fl,fu)