import torch.nn.functional as F
import torch
from torch.optim import AdamW
from utils import now_time, root_mean_square_error, mean_absolute_error


def train_rating_predictor(model, train_data, test_data, val_data, model_path, lr=1e-3, max_epoch=20, patience=3, max_rating=5.0, min_rating=1.0, device='cpu'):
    print(now_time() + 'Training Rating Predictor.')

    # Set the weights to be updated
    for name, param in model.named_parameters():
        if 'f_r' in name or 'f_user' in name or 'f_item' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    rating_params = list(model.f_r.parameters()) + list(model.f_user.parameters()) + list(model.f_item.parameters())
    optimizer_rating = AdamW(rating_params, lr=lr, weight_decay=1e-3)

    best_rmse = float('inf')
    patience_counter = 0

    for epoch in range(1, max_epoch + 1):
        print(now_time() + 'epoch {}'.format(epoch))
    
        model.train()
        rating_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, *_ = train_data.next_batch() 
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)

            optimizer_rating.zero_grad()
            rating_p = model.predict_rating(user, item) 
            loss = F.mse_loss(rating_p.to(torch.float32), rating.to(torch.float32))
            loss.backward()
            optimizer_rating.step()

            batch_size = user.size(0)
            rating_loss += batch_size * loss.item()
            total_sample += batch_size
            if train_data.step == train_data.total_step:
                cur_t_loss = rating_loss / total_sample
                print(now_time() + 'loss {:4.4f}'.format(cur_t_loss))
                break
        
        # Evaluate the model on the validation set
        rating_predicted = predict_rating(model, val_data, device=device)
        predicted_rating = [(r * max_rating, p * max_rating) for (r, p) in zip(val_data.rating.tolist(), rating_predicted)]
        rmse = root_mean_square_error(predicted_rating, max_rating, min_rating)
        print(now_time() + 'Validation RMSE {:7.4f}'.format(rmse))

        rating_predicted = predict_rating(model, test_data, device=device)
        predicted_rating = [(r * max_rating, p * max_rating) for (r, p) in zip(test_data.rating.tolist(), rating_predicted)]
        RMSE = root_mean_square_error(predicted_rating, max_rating, min_rating)
        print(now_time() + 'Test RMSE {:7.4f}'.format(RMSE))
        MAE = mean_absolute_error(predicted_rating, max_rating, min_rating)
        print(now_time() + 'Test MAE {:7.4f}'.format(MAE))
        
        
        # Early stop mechanism
        if rmse < best_rmse:
            best_rmse = rmse
            patience_counter = 0
            trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(trainable_state_dict, model_path + "_rating.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:  # stop training
                print(now_time() + 'Early stopping triggered. No improvement in validation loss.')
                print(now_time() + 'Loading best rating predictor weights...')
                model.load_state_dict(torch.load(model_path + "_rating.pt", map_location=device), strict=False)
                break


def predict_rating(model, data, device='cpu'):
    model.eval()
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, *_ = data.next_batch() 
            user = user.to(device)
            item = item.to(device)

            rating_p = model.predict_rating(user, item) 
            rating_predict.extend(rating_p.tolist())

            if data.step == data.total_step:
                break
    return rating_predict