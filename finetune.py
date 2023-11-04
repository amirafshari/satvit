from utils.finetuning import TrainingPipeline
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
    parser.add_argument('--config_file', type=str, default="configs/Finetuning.yaml",
                        help='.yaml configuration file to use')

    opt = parser.parse_args()
    config_file = opt.config_file

    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    architecture = config['MODEL']['architecture']
    batch_size = config['TRAIN']['batch_size']
    learning_rate = config['TRAIN']['learning_rate']
    max_epochs = config['TRAIN']['max_epochs']
    dataset_path = config['DATA']['Pastis24']
    train_ratio = config['DATA']['train_ratio']
    val_ratio = config['DATA']['val_ratio']

    pipeline = TrainingPipeline(architecture, dataset_path, batch_size, train_ratio, val_ratio, learning_rate, max_epochs)
    
    # tb_process = start_tensorboard()
    # print("TensorBoard started. You can view logs at http://localhost:6006/")

    pipeline.run()

    # tb_process.terminate()
    # print("TensorBoard terminated.")



if __name__ == "__main__":
    main()

