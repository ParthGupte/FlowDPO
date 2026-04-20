from torch import nn
import torch

class FlowDPOLoss(nn.Module):
    def __init__(self, beta = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(self,delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,eps,y_t5_l,y_t5_w):
        term_1 = (delta_t1_w*(2*y_w-delta_t2_w)).sum()
        term_2 = (-delta_t3_l*(2*y_l-delta_t4_l)).sum()
        s = ((delta_t5_l-delta_t5_w)*eps).sum()
        grad_l,grad_w = torch.autograd.grad(s,(y_t5_l,y_t5_w),create_graph=True)
        term_3 = (eps*(grad_l+grad_w)).sum()
        return term_1+term_2+term_3 


