import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    #SR=SR.int()
    #GT=GT.int()
    SR1=(SR==1).int()
    GT1=(GT==1).int()
    SR0= (SR==0).int()
    GT0 = (GT == 0).int()
    # TP : True Positive
    # FN : False Negative
    TP= torch.sum((SR1+GT1)==2)
    FN= torch.sum((SR0+GT0)==2)

    #TP =torch.sum(SR1GT)
    # ((SR==1)+(GT==1))==2
    #FN = ((SR==0)+(GT==1))==2
    #FN = torch.sum(SR0==GT)
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR1 = (SR == 1).int()
    GT1 = (GT == 1).int()
    SR0 = (SR == 0).int()
    GT0 = (GT == 0).int()
    # TP : True Positive
    # FN : False Negative
    TN = torch.sum((SR0 + GT0) == 2)
    FP = torch.sum((SR1 + GT0) == 2)
    # TN : True Negative
    # FP : False Positive
    #TN= torch.sum(SR0==GT0)
    #FP= torch.sum(SR==GT0)
    #TN = ((SR==0)+(GT==0))==2
    #FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR1 = (SR == 1).int()
    GT1 = (GT == 1).int()
    SR0 = (SR == 0).int()
    GT0 = (GT == 0).int()
    # TP : True Positive
    # FN : False Negative
    TP = torch.sum((SR1 + GT1) == 2)
    FP = torch.sum((SR1 + GT0) == 2)
    # TP : True Positive
    # FP : False Positive
    #TP= torch.sum(SR==GT)
    #FP=torch.sum(SR==GT0)
    #TP = ((SR==1)+(GT==1))==2
    #FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR1 = (SR == 1).int()
    GT1 = (GT == 1).int()
    Inter = torch.sum((SR1+GT1)==2)
    Union = torch.sum((SR1+GT1)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR1 = (SR == 1).int()
    GT1 = (GT == 1).int()
    Inter = torch.sum((SR1+GT1)==2)
    DC = float(2*Inter)/(float(torch.sum(SR1)+torch.sum(GT1)) + 1e-6)

    return DC



