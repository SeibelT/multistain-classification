from tqdm import tqdm
import numpy as np
import torch

def trainer(model,n_epochs,train_loader,valid_loader,scheduler,device,criterion,optimizer,writer,auroc,activate,unfreeze,epoch_checkpoint=50,checkpoint_path="results/"):
    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_auroc = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for idx,(x, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}|Train")

                if unfreeze==idx:
                    for param in model.parameters():
                        param.requires_grad = True
                    print("Unfreeze feature extractors\n")

                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred,y.float())
                loss.backward()
                optimizer.step()
                
                tr_auroc = auroc(activate(pred), y.int())

                train_loss.append(loss.item())
                train_auroc.append(tr_auroc)
                tepoch.set_postfix(train_loss=loss.item(), train_AUROC=tr_auroc.item())

                writer.add_scalar('training_loss', loss.item(), global_step=epoch * len(train_loader) + idx)
                writer.add_scalar('training_auroc', tr_auroc.item(), global_step=epoch * len(train_loader) + idx)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, global_step=epoch * len(train_loader) + idx)

                
                if idx>0:  # get rid of before running 
                    break
        if epoch%epoch_checkpoint==0:
            """store checkpoints every epoch_checkpoint """
            path = checkpoint_path + f"/model_at_ep{epoch}.pt"
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_mean_loss': np.mean(train_loss),
            }, path)

        scheduler.step()
        valid_loss = []
        valid_auroc = []
        model.eval()
        with tqdm(valid_loader, unit="batch") as vepoch:
            for idx,(x, y) in enumerate(vepoch):
                vepoch.set_description(f"Epoch {epoch+1}|Valid")
                
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                
                loss= criterion(out,y.float())
                val_auroc  = auroc(activate(out), y.int())
                
                valid_loss.append(loss.item())
                valid_auroc.append(val_auroc)
                vepoch.set_postfix(valid_loss=loss.item(), valid_AUROC=val_auroc.item())

                writer.add_scalar('validation_loss', loss.item(), global_step=epoch * len(valid_loader) + idx)
                writer.add_scalar('validation_auroc', val_auroc.item(), global_step=epoch * len(valid_loader) + idx)


                if idx>0: # TODO get rid of before running 
                    break

        
        

    return model,optimizer,epoch




