import torch
from tqdm import tqdm


def evaluation(model,test_loader,device,criterion,auroc,activate):
    test_loss = []
    test_aurocs = []
    model.eval()
    results = []
    with tqdm(test_loader, unit="batch") as testepoch:
        for idx,(x, y) in enumerate(testepoch):
            testepoch.set_description(f"Test")
            x=x.to(device)
            y=y.to(device)
            out = model(x)
            loss = criterion(out,y.float())
            test_auroc = auroc(activate(out), y.int())

            results.append( torch.cat((y.float(),activate(out)),dim=1) )
            
            test_loss.append(loss.item())
            test_aurocs.append(test_auroc.item())
            testepoch.set_postfix(test_loss=loss.item(), test_AUROC=test_auroc.item())


            if idx>0: # TODO get rid of before running 
                break

        #mean_test_loss = np.mean(test_loss)  # TODO save in tensorboard
        #mean_test_auroc = np.mean(test_aurocs)  # TODO save in tensorboard
        return torch.cat(results).cpu().detach().numpy()
    