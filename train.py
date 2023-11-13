from utils.training import TrainingPipeline
import time
import yaml
import argparse
import subprocess
import webbrowser





def start_tensorboard(log_dir='./tb_logs', port=6006):
    cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(cmd, shell=True)
    webbrowser.open(f'http://localhost:{port}/')
    return process

def main():

    parser = argparse.ArgumentParser(description='Enter your parameters')
    parser.add_argument('--config_file', type=str, default="configs/segmentation.yaml",
                        help='.yaml configuration file to use')
    parser.add_argument('--resume', type=str, default=False)

    opt = parser.parse_args()
    config_file = opt.config_file
    resume = opt.resume
    

    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    architecture = config['MODEL']['architecture']
    img_width = config['MODEL']['img_width']
    img_height = config['MODEL']['img_height']
    in_channel = config['MODEL']['in_channel']
    patch_size = config['MODEL']['patch_size']
    embed_dim = config['MODEL']['embed_dim']
    max_time = config['MODEL']['max_time']
    num_classes = config['MODEL']['num_classes']
    num_head = config['MODEL']['num_head']
    dim_feedforward = config['MODEL']['dim_feedforward']
    num_layers = config['MODEL']['num_layers']
    batch_size = config['TRAIN']['batch_size']
    learning_rate = config['TRAIN']['learning_rate']
    max_epochs = config['TRAIN']['max_epochs']
    dataset_path = config['DATA']['Pastis24']
    train_ratio = config['DATA']['train_ratio']
    val_ratio = config['DATA']['val_ratio']
    dropoutratio = config['MODEL']['dropoutratio']
    l2 = config['TRAIN']['l2']

    pipeline = TrainingPipeline(architecture, dataset_path, batch_size, train_ratio, val_ratio, learning_rate, max_epochs, img_width, img_height, in_channel, patch_size, embed_dim, max_time, num_classes, num_head, dim_feedforward, num_layers, dropoutratio)
    
    # tb_process = start_tensorboard()
    # print("TensorBoard started. You can view logs at http://localhost:6006/")
    if resume:
        pipeline.resume_training(resume)
        
    pipeline.run()

    # tb_process.terminate()
    # print("TensorBoard terminated.")



if __name__ == "__main__":
    main()

