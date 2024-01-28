import os
import datetime
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.nn as nn

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

    def __init__(self, architecture ,dataset_path, batch_size, train_ratio, val_ratio, learning_rate, max_epochs, img_width, img_height, in_channel, patch_size, embed_dim, max_time, num_classes, num_head, dim_feedforward, num_layers, dropoutratio, l2):
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
        self.l2 = l2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.architecture == "segmentation":
            self.model = Segmentation(img_width=img_width, img_height=img_height, in_channel=in_channel,
                            patch_size=patch_size, embed_dim=embed_dim, max_time=max_time,
                            num_head=num_head, num_layers=num_layers,
                            num_classes=num_classes, dropoutratio=dropoutratio)
            self.model= nn.DataParallel(self.model)
            self.model.to(self.device)
            self.criterion = MaskedCrossEntropyLoss()



        elif self.architecture == "classification":
            self.model = Classification(img_height=img_height, img_width=img_width, in_channel=in_channel,
                       patch_size=patch_size, embed_dim=embed_dim, max_time=max_time,
                       num_classes=num_classes, num_head=num_head, dim_feedforward=dim_feedforward,
                       num_layers=num_layers, dropoutratio=dropoutratio)
            self.model= nn.DataParallel(self.model)
            self.model.to(self.device)
            self.criterion = FocalLoss()



        self.data_loader = PASTISDataLoader(dataset_path=dataset_path, batch_size=batch_size, train_ratio=train_ratio, val_ratio=val_ratio)
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.get_data_loaders()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger('ignite.engine.engine.Engine').setLevel(logging.WARNING)





    @staticmethod
    def setup_directories(architecture):
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(f'runs/{architecture}/'):
            os.makedirs(f'runs/{architecture}/')
        if not os.path.exists(f'runs/{architecture}/'):
            os.makedirs(f'runs/{architecture}/')


        if not os.path.exists(f'runs/{architecture}/{date}/logs'):
            os.makedirs(f'runs/{architecture}/{date}/logs/')
        if not os.path.exists(f'runs/{architecture}/{date}/weights/'):
            os.makedirs(f'runs/{architecture}/{date}/weights/')

        return date

    def train_model(self, dir):

        # WandBlogger Object Creation
        wandb_logger = WandBLogger(
        project="satellite-time-series",
        config={"max_epochs": self.max_epochs, "batch_size": self.batch_size, 'learning_rate': self.learning_rate,
                'num_head': self.num_head, 'dim_feedforward': self.dim_feedforward, 'num_layers': self.num_layers,
                'dropoutratio': self.dropoutratio, 'l2': self.l2}, 
        tags=["pastis", self.architecture]
        )
        


        # Metrics
        val_metrics = {
            "accuracy": Accuracy(),
            # "precision": Precision(),
            # "recall": Recall(),
            # "confusion_matrix": ConfusionMatrix(num_classes=20),
            "loss": Loss(self.criterion)
        }


        # Trainer
        def train_step(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        trainer = Engine(train_step)



        # Evaluator
        def validation_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                y_pred = self.model(x)
                return y_pred, y
        evaluator = Engine(validation_step)




        # Attach metrics to the evaluators
        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)











        ''' Logs '''


        # Add progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, output_transform=lambda loss: {'iteration loss': loss}, event_name=Events.ITERATION_COMPLETED, closing_event_name=Events.EPOCH_COMPLETED)

        # Set up Tensorboard logging
        tb_logger = TensorboardLogger(log_dir=f'./runs/{self.architecture}/{dir}/logs/')
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}), event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=["loss", "accuracy"], global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        # Checkpoint Handler
        checkpoint_handler = ModelCheckpoint(f'./runs/{self.architecture}/{dir}/weights//', 'model', n_saved=3, require_empty=False)





        # Early stopping
        # handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        # evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
        # Learning rate scheduler
        # scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        # trainer.add_event_handler(Events.EPOCH_COMPLETED, LRScheduler(scheduler))








        # Save the last model after each epoch
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_last_model(engine):
            epoch = engine.state.epoch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            os.path.join(f'./runs/{self.architecture}/{dir}/weights/', 'last_model.pth'))




        # Save the best model based on validation metrics
        best_accuracy = 0.0
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_best_model_validation_metrics(engine):

            # Metrics
            evaluator.run(self.val_loader)
            metrics = evaluator.state.metrics
            validation_loss = metrics['loss']
            validation_accuracy = metrics['accuracy']




            # Validation Metrics
            wandb_logger.attach_output_handler(
            evaluator,
            tag="validation",
            event_name=Events.EPOCH_COMPLETED,
            output_transform=lambda x: {'Validation Epoch Accuracy': validation_accuracy, 'Validation Epoch Loss': validation_loss},
            global_step_transform=lambda *_: trainer.state.epoch
            )


            # Progress Bar
            pbar.log_message(f"Validation Results - Epoch: {engine.state.epoch}  Validation Epoch Loss: {validation_loss:.4f} Validation Epoch Accuracy: {validation_accuracy:.4f}")



            # Save checkpoint
            nonlocal best_accuracy
            if validation_accuracy > best_accuracy:
                epoch = engine.state.epoch
                torch.save({
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(f'./runs/{self.architecture}/{dir}/weights/', f'best_epoch_{epoch}_val_acc_{validation_accuracy}.pth'))
                best_accuracy = validation_accuracy





        # Log Training Metrics
        @trainer.on(Events.EPOCH_COMPLETED(every=10))
        def log_training_metrics(engine):
            evaluator.run(self.train_loader)
            metrics = evaluator.state.metrics
            training_loss = metrics['loss'] # you can comment it
            training_accuracy = metrics['accuracy']

            # Progress Bar
            pbar.log_message(f"Training Results - Epoch: {engine.state.epoch} Training Epoch Accuracy: {training_accuracy:.4f} Training Epoch Loss: {training_loss:.4f}")



            # Training Metrics every N epoch
            wandb_logger.attach_output_handler(
            evaluator,
            tag="training",
            event_name=Events.EPOCH_COMPLETED,
            output_transform=lambda x: {'Training Epoch Accuracy': training_accuracy, 'Training Epoch Loss': training_loss},
            global_step_transform=lambda *_: trainer.state.epoch
            )




        # Training Loss
        wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        output_transform=lambda loss: {'loss': loss},
        )

        trainer.run(self.train_loader, max_epochs=self.max_epochs)
        wandb_logger.close()
        tb_logger.close()

    def run(self):
        dir = self.setup_directories(self.architecture)
        self.train_model(dir)

    def resume_training(self, checkpoint_path):
        # Load the model state from the checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.run()