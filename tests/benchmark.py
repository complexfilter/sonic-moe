import torch
from sonicmoe import MoE, KernelBackendMoE
import triton 
from dcmoe.mlp import MLP_SiWGLU
import torch.nn.functional as F
# Create MoE layer

moe = MoE(
    num_experts=64,           # Number of experts
    num_experts_per_tok=7,     # Top-k experts per token
    hidden_size=2048,          # Hidden dimension
    intermediate_size=1024,    # Expert intermediate size
    is_glu=True,               # Whether to use GLU (e.g. SwiGLU) activation
    add_bias=False,            # Add bias to linear layers
    std=0.02,                  # Weight initialization std
).to(device="cuda", dtype=torch.bfloat16)

# Forward pass
torch.manual_seed(43)
x = torch.randn(8000, 2048, device="cuda", dtype=torch.bfloat16)
output = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)

def fwd_bwd_sonic(x, moe, dY):
    output = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    dX, dW, dW_out = torch.autograd.grad(
        outputs=(output,),
        inputs=(x, moe.c_fc.weight, moe.c_proj.weight),
        grad_outputs=(dY,)
    )
    return output, dX, dW, dW_out

mlp = MLP_SiWGLU(
    input_size=2048, hidden_size=1024,
    num_experts=64, top_k=7
).cuda().to(torch.bfloat16)


with torch.no_grad():
    mlp.experts_w1.weight.copy_(moe.c_fc.weight[:, ::2, :])
    mlp.experts_w3.weight.copy_(moe.c_fc.weight[:, 1::2, :])
    mlp.output_experts.weight.copy_(moe.c_proj.weight)

def MLP(x, mlp, num_activated_experts):
    with torch.no_grad():
        logits = F.linear(x, moe.router.weight)
        gates, k_idxs = torch.topk(logits, num_activated_experts)
        gates = torch.softmax(gates.float(), axis=-1).cuda().to(torch.bfloat16)
    Y = mlp(x, gates, k_idxs)
    return Y


def fwd_bwd_dcmoe(x, mlp, num_activated_experts, dY):
    Y = MLP(x, mlp, num_activated_experts)
    dX, dW1, dW3, dW_out = torch.autograd.grad(
        outputs=(Y,),
        inputs=(x, mlp.experts_w1.weight, mlp.experts_w3.weight, mlp.output_experts.weight),
        grad_outputs=(dY,)
    )
    return Y, dX, dW1, dW3, dW_out

Y = MLP(x, mlp, 7)
dY = torch.randn_like(x)

relative_error = torch.norm(Y - output, p='fro') / torch.norm(output, p='fro')
print("relative error:", relative_error.item())
print(output.shape)

ms = triton.testing.do_bench(lambda: MLP(x, mlp, 7), warmup=25, rep=100, quantiles = [0.2,0.5,0.8])
print("runtime of dcmoe forward is:", ms[1])

ms = triton.testing.do_bench(lambda: moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe), warmup=25, rep=100, quantiles = [0.2,0.5,0.8])
print("runtime of sonic moe forward is:", ms[1])

x = x.detach().requires_grad_(True)   # ens
ms = triton.testing.do_bench(lambda: fwd_bwd_dcmoe(x, mlp, 7, dY), warmup=25, rep=100, quantiles = [0.2,0.5,0.8])
print("runtime of fwd and bwd of dcmoe is:", ms[1])

ms = triton.testing.do_bench(lambda: fwd_bwd_sonic(x, moe, dY), warmup=25, rep=100, quantiles = [0.2,0.5,0.8])
print("runtime of fwd and bwd of sonic moe is:", ms[1])
