from torch import nn
import torch
import torch.nn.functional as F

class FlowDPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
    
    def forward(self,delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,y_t5_l,y_t5_w):
        B = y_w.shape[0]

        term_1 = (delta_t1_w*(2*y_w-delta_t2_w)).view(B,-1).mean(dim=1)
        term_2 = (-delta_t3_l*(2*y_l-delta_t4_l)).view(B,-1).mean(dim=1)

        eps = torch.randn_like(y_w)

        s_l = (delta_t5_l * eps).view(B,-1).sum(dim=1)
        s_w = (delta_t5_w * eps).view(B,-1).sum(dim=1)

        grad_l = torch.autograd.grad(
            s_l,
            y_t5_l,
            grad_outputs=torch.ones_like(s_l),
            create_graph=True
        )[0]

        grad_w = torch.autograd.grad(
            s_w,
            y_t5_w,
            grad_outputs=torch.ones_like(s_w),
            create_graph=True
        )[0]

        term_3 = (2*eps*(grad_l - grad_w)).view(B,-1).mean(dim=1)

        return -F.logsigmoid((self.beta/2)*(term_1 + term_2 + term_3)).mean()

class FlowDPOLossMultiSample(nn.Module):
    def __init__(self, beta=0.1,N=100):
        super().__init__()
        self.beta = beta
        self.N = N
    
    def forward(self,delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,y_t5_l,y_t5_w):
        B, C, H, W = y_w.shape

        term_1 = (delta_t1_w*(2*y_w-delta_t2_w)).view(B,-1).mean(dim=1)
        term_2 = (-delta_t3_l*(2*y_l-delta_t4_l)).view(B,-1).mean(dim=1)

        eps = torch.randn((B,self.N,C*H*W),device=y_w.device)

        s_l = torch.matmul(eps,delta_t5_l.view(B,-1).unsqueeze(-1)).squeeze(-1).sum(dim=1)
        s_w = torch.matmul(eps,delta_t5_w.view(B,-1).unsqueeze(-1)).squeeze(-1).sum(dim=1)

        grad_l = torch.autograd.grad(
            s_l,
            y_t5_l,
            grad_outputs=torch.ones_like(s_l),
            create_graph=True
        )[0]

        grad_w = torch.autograd.grad(
            s_w,
            y_t5_w,
            grad_outputs=torch.ones_like(s_w),
            create_graph=True
        )[0]

        term_3 = torch.matmul(eps,(grad_l - grad_w).view(B,-1).unsqueeze(-1)).squeeze(-1).mean(dim=1)

        return -F.logsigmoid((self.beta/2)*(term_1 + term_2 + 2*term_3)).mean()