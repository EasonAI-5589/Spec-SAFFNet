import os
import pickle

def save_model_info(name, path, model, train_rmse, train_r2, test_rmse, test_r2,dataset_dict):
    model_info = {
        'model_state_dict': model.state_dict(),
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'dataset_dict': dataset_dict
    }
    with open(os.path.join(path, f'model_info_{name}.pkl'), 'wb') as f:
        pickle.dump(model_info, f)

