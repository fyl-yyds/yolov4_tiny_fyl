import torch
def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

# b = torch.randint(0,10,(2,2))
# print(b)
# min = torch.randint(0,6,(2,2))
# print(min)
# max = torch.randint(5,10,(2,2))
# print(max)
# a = clip_by_tensor(b,min,max)
# print(a)

a=torch.zeros(3,5)
b=torch.zeros(len(a),5)
print(b.shape)
print(b[:len(b)])