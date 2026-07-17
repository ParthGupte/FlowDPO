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
    def __init__(self, beta=0.1,N=1000):
        super().__init__()
        self.beta = beta
        self.N = N
    
    def forward(self,delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,y_t5_l,y_t5_w):
        B, C, H, W = y_w.shape

        term_1 = (delta_t1_w*(2*y_w-delta_t2_w)).view(B,-1).mean(dim=1)
        term_2 = (-delta_t3_l*(2*y_l-delta_t4_l)).view(B,-1).mean(dim=1)

        eps = torch.randn((B,self.N,C*H*W),device=y_w.device,dtype=y_w.dtype)

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

class FlowDPOLossMultiSampleRademacher(nn.Module):
    def __init__(self, beta=0.1,N=1000):
        super().__init__()
        self.beta = beta
        self.N = N
    
    def forward(self,delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,y_t5_l,y_t5_w):
        B, C, H, W = y_w.shape

        term_1 = (delta_t1_w*(2*y_w-delta_t2_w)).view(B,-1).mean(dim=1)
        term_2 = (-delta_t3_l*(2*y_l-delta_t4_l)).view(B,-1).mean(dim=1)

        eps = (torch.randint(0,1,(B,self.N,C*H*W),device=y_w.device)-0.5)*2

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
    
class FlowDPOLossWallace(nn.Module):
    def __init__(self, beta=0.1,N=1000):
        super().__init__()
        self.beta = beta
        self.N = N
    
    def forward(self,V_theta,V_ref,V_w,V_l):
        B, C, H, W = V_theta.shape

        term_1 = ((V_w-V_theta)**2).view(B,-1).mean(dim=1) - ((V_l-V_theta)**2).view(B,-1).mean(dim=1)
        term_2 = ((V_l-V_ref)**2).view(B,-1).mean(dim=1) - ((V_w-V_ref)**2).view(B,-1).mean(dim=1)

        

        return -F.logsigmoid((self.beta/2)*(term_1 - term_2)).mean()


class FlowDPOLossMultiSampleOrtho(nn.Module):
    """Multi-sample DPO loss that uses orthogonal sample vectors for eps.

    eps is sampled as Gaussian and orthonormalized via QR decomposition per batch.
    If `N` (number of samples) is greater than the data dimension `D = C*H*W`,
    orthogonal sampling is not possible and the implementation falls back to
    standard Gaussian sampling with a warning.
    """
    def __init__(self, beta=0.1, N=1000):
        super().__init__()
        self.beta = beta
        self.N = N

    def forward(self,delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,y_t5_l,y_t5_w):
        B, C, H, W = y_w.shape

        term_1 = (delta_t1_w*(2*y_w-delta_t2_w)).view(B,-1).mean(dim=1)
        term_2 = (-delta_t3_l*(2*y_l-delta_t4_l)).view(B,-1).mean(dim=1)

        D = C*H*W

        # Sample orthogonal vectors via batched QR. Require N <= D.
        if self.N > D:
            raise ValueError(f"FlowDPOLossMultiSampleOrtho requires N <= D (got N={self.N}, D={D})")

        # Sample gaussian matrix of shape (B, D, N) then QR -> Q: (B, D, N)
        A = torch.randn((B, D, self.N), device=y_w.device, dtype=y_w.dtype)
        Q, R = torch.linalg.qr(A, mode='reduced')
        # Q has shape (B, D, N), we need eps as (B, N, D)
        eps = Q.transpose(1, 2).contiguous()

        # compute projections
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