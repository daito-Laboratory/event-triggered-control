
import os
import sys
import torch
import parameter
import matplotlib.pyplot as plt
import importlib.util
# my_module のパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from My_module import my_function_ET
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plant
from plant import Plant_over
import importlib.util
module_path = os.path.join(os.getcwd(),parameter.Folderpath, "result.py")
# モジュールのspecをロード
spec = importlib.util.spec_from_file_location("result", module_path)
result = importlib.util.module_from_spec(spec)
# モジュールを実行してロード
spec.loader.exec_module(result)
K_best = torch.tensor(result.Best_K, dtype=torch.double)

num_hidden_best = int(result.Best_num_hidden)
node_hidden_best = int(result.Best_node_hidden)

hidden_W_best = [
    torch.tensor(w, dtype=torch.double) for w in result.Best_hidden_weights
]
hidden_b_best = [
    torch.tensor(b, dtype=torch.double) for b in result.Best_hidden_biases
]

out_W_best = torch.tensor(result.Best_out_weight, dtype=torch.double)
out_b_best = torch.tensor(result.Best_out_bias, dtype=torch.double)
# my_module のモジュールをインポート
T= plant.T
h=plant.h
N=plant.N
M= plant.M
systemtype= plant.system_type
epoch = parameter.epoch
lr = parameter.lr
path = parameter.Folderpath
min_val = parameter.min_val
max_val = parameter.max_val
lam = parameter.lam
Batch = parameter.Batch
datasize = parameter.datasize
AD = parameter.AD
evolution= parameter.evol
Switch = parameter.Switch
Q=plant.Q
Qf= plant.Qf
R= plant.R
Plant=Plant_over(h=h)
cstr = my_function_ET.CSTR(T,h,N,M,Q,Qf,R,Plant,systemtype,epoch,lr,path,min_val,max_val,lam,Batch,datasize,AD,evolution,Switch)
hidden_layer=parameter.hidden_layer
hidden_node=parameter.hidden_node
total_cost_Best = 0
total_event_Best = 0
my_function_ET.seedinit(1)
x0 = my_function_ET.Mike_x0(min_val, max_val, evolution, N)
for i in range(evolution):
    cost, event = my_function_ET.Evolute_NN(cstr,
                                                T, x0[i], i,K_best, hidden_W_best,hidden_b_best,out_W_best,out_b_best, N, M, Q, R,num_hidden_best,node_hidden_best, path)
    total_cost_Best += cost
    total_event_Best += event
Average_cost_Best = total_cost_Best/evolution
Average_event_Best = total_event_Best/evolution
print("Average_cost", Average_cost_Best)
print("Average_event", Average_event_Best)
# # TODO----------------------------------------------------------------検証2

# total_cost = 0
# total_event = 0
# for i in range(evolution):
#     cost, event = my_function_ET.Evolute_4para2(cstr,
#                                                 T, x0[i], i, K_b, a_b, b_b, c_b, N, M, Q, R, path)
#     total_cost += cost
#     total_event += event
# Average_cost = total_cost/evolution
# Average_event = total_event/evolution
# print("Average_cost", Average_cost)
# print("Average_event", Average_event)

plt.show()