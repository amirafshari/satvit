import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import logging

from ignite.metrics import Metric, Loss, Accuracy
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar  
from ignite.handlers import ModelCheckpoint, EarlyStopping, LRScheduler
from ignite.contrib.handlers.wandb_logger import *
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, global_step_from_engine

from utils.dataset import PASTIS, PASTISDataLoader
from criterion.loss import MaskedCrossEntropyLoss , FocalLoss
from criterion.accuracy import CustomAccuracy
from models.segmentation import Segmentation 
from models.classification import Classification



class TrainingPipeline:

    def __init__(self, architecture ,dataset_path, batch_size, train_ratio, val_ratio, learning_rate, max_epochs, img_width, img_height, in_channel, patch_size, embed_dim, max_time, num_classes, num_head, dim_feedforward, num_layers, dropoutratio ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.img_width = img_width
        self.img_height = img_height
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_time = max_time
        self.num_classes = num_classes
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.architecture = architecture
        self.dropoutratio = dropoutratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.architecture == "segmentation":
            self.model = Segmentation(img_width=img_width, img_height=img_height, in_channel=in_channel,
                            patch_size=patch_size, embed_dim=embed_dim, max_time=max_time,
                            num_head=num_head, num_layers=num_layers,
                            num_classes=num_classes, dropoutratio=dropoutratio).to(self.device)
            self.criterion = MaskedCrossEntropyLoss()



        elif self.architecture == "classification":
            self.model = Classification(img_height=img_height, img_width=img_width, in_channel=in_channel,
                       patch_size=patch_size, embed_dim=embed_dim, max_time=max_time,
                       num_classes=num_classes, num_head=num_head, dim_feedforward=dim_feedforward,
                       num_layers=num_layers, dropoutratio=dropoutratio).to(self.device)
            self.criterion = FocalLoss()



        self.data_loader = PASTISDataLoader(dataset_path=dataset_path, batch_size=batch_size, train_ratio=train_ratio, val_ratio=val_ratio)
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.get_data_loaders()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger('ignite.engine.engine.Engine').setLevel(logging.WARNING)





    @staticmethod
    def setup_directories(log_dir, model_save_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

    def train_model(self):

        #WandBlogger Object Creation
        wandb_logger = WandBLogger(
        project="pytorch-ignite-integration",
        name="classification",
        config={"max_epochs": self.max_epochs, "batch_size": self.batch_size},
        tags=["pastis", "classification"]
        )
        
        # Create trainer and evaluator
        val_metrics = {
        "Accuracy": CustomAccuracy(),
        "Loss": Loss(self.criterion)
        }
        trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion, device=self.device)
        evaluator = create_supervised_evaluator(self.model, metrics=val_metrics, device=self.device)


        # Logs
        wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss}
        )

        wandb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["Loss", "Accuracy"],
        global_step_transform=lambda *_: trainer.state.epoch
        )

        # Add progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})


        # Set up Tensorboard logging
        tb_logger = TensorboardLogger(log_dir=f'./tb_logs_{self.architecture}')
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}), event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=["Loss", "Accuracy"], global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)


        # Checkpoint Handler
        checkpoint_handler = ModelCheckpoint(f'./checkpoints_{self.architecture}', 'model', n_saved=3, require_empty=False)


        # Early stopping
        def score_function(engine):
            val_loss = engine.state.metrics['Loss']
            return -val_loss

        handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

        # Learning rate scheduler
        scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, LRScheduler(scheduler))


        # Save the last model after each epoch
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_last_model(engine):
            epoch = engine.state.epoch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            os.path.join(f'./checkpoints_{self.architecture}', 'last_model.pth'))

        # Save the best model based on validation accuracy
        best_accuracy = 0.0
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_best_model(engine):
            evaluator.run(self.val_loader)
            accuracy = evaluator.state.metrics['Accuracy']
            nonlocal best_accuracy
            if accuracy > best_accuracy:
                epoch = engine.state.epoch
                torch.save({
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(f'./checkpoints_{self.architecture}', f'best_model_epoch_{epoch}.pth'))
                best_accuracy = accuracy

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.model.eval()
            evaluator.run(self.train_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics['Loss']
            avg_accuracy = metrics['Accuracy']
            pbar.log_message(f"Training Results - Epoch: {engine.state.epoch}  Avg loss: {avg_loss:.4f} Avg accuracy: {avg_accuracy:.4f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.model.eval()
            evaluator.run(self.val_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics['Loss']
            avg_accuracy = metrics['Accuracy']
            pbar.log_message(f"Validation Results - Epoch: {engine.state.epoch}  Avg loss: {avg_loss:.4f} Avg accuracy: {avg_accuracy:.4f}")

        trainer.run(self.train_loader, max_epochs=self.max_epochs)
        wandb_logger.closer()
        tb_logger.close()

    def run(self):
        self.setup_directories(f'./tb_logs_{self.architecture}', f'./checkpoints_{self.architecture}')
        self.train_model()



    def resume_training(self, checkpoint_path):
        # Load the model state from the checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.run()