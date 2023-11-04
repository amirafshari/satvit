import os
import torch
import torch.nn.functional as F
import logging
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Loss
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, WeightsScalarHandler, WeightsHistHandler, GradsScalarHandler
from ignite.handlers import LRScheduler
from torch.optim.lr_scheduler import StepLR
from utils.dataset import PASTIS, PASTISDataLoader
from criterion.loss import MaskedCrossEntropyLoss , FocalLoss
from models.segmentation import Segmentation 
from models.classification import Classification
from torch import optim
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from criterion.accuracy import CustomAccuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar  
import torch as nn


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

        if self.architecture == "segmentation":
            
            self.model = Segmentation(img_width=img_width, img_height=img_height, in_channel=in_channel, patch_size=patch_size, embed_dim=embed_dim, max_time=max_time, num_head=num_head, num_layers=num_layers, num_classes=num_classes, dropoutratio=dropoutratio)
            self.criterion = MaskedCrossEntropyLoss()
        elif self.architecture == "classification":

            self.model = Classification(img_height=img_height, img_width=img_width, in_channel=in_channel,
                       patch_size=patch_size, embed_dim=embed_dim, max_time=max_time,
                       num_classes=num_classes, num_head=num_head, dim_feedforward=dim_feedforward,
                       num_layers=num_layers, dropoutratio=dropoutratio)
            
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Create trainer and evaluator
        trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion, device=device)
        evaluator = create_supervised_evaluator(self.model, metrics={'Loss': Loss(self.criterion), 'Accuracy': CustomAccuracy()}, device=device)
        
        @trainer.on(Events.EPOCH_STARTED)
        def start_of_epoch(engine):
            self.model.train() # to emsure dropout works correctly
            print(f"Starting Epoch {engine.state.epoch}...")
        # Add progress bar

        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})

        # Set up Tensorboard logging
        tb_logger = TensorboardLogger(log_dir=f'./tb_logs_{self.architecture}')

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=WeightsScalarHandler(self.model), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=WeightsHistHandler(self.model), event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(trainer, log_handler=GradsScalarHandler(self.model), event_name=Events.ITERATION_COMPLETED)


        checkpoint_handler = ModelCheckpoint(f'./checkpoints_{self.architecture}', 'model', n_saved=3, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': self.model})


        # Early stopping
        def score_function(engine):
            val_loss = engine.state.metrics['Loss']
            return -val_loss

        handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

        # Learning rate scheduler
        scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, LRScheduler(scheduler))


        # Save the first model after the first epoch
        @trainer.on(Events.EPOCH_COMPLETED(once=1))
        def save_first_model(engine):
            epoch = engine.state.epoch
            torch.save(self.model.state_dict(), os.path.join(f'./checkpoints_{self.architecture}', 'first_model.pth'))

        # Save the last model after each epoch
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_last_model(engine):
            epoch = engine.state.epoch
            torch.save(self.model.state_dict(), os.path.join(f'./checkpoints_{self.architecture}', 'last_model.pth'))

        # Save the best model based on validation accuracy
        best_accuracy = 0.0
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_best_model(engine):
            evaluator.run(self.val_loader)
            accuracy = evaluator.state.metrics['Accuracy']
            nonlocal best_accuracy
            if accuracy > best_accuracy:
                epoch = engine.state.epoch
                torch.save(self.model.state_dict(), os.path.join(f'./checkpoints_{self.architecture}', f'best_model_epoch_{epoch}.pth'))
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
        tb_logger.close()

    def run(self):
        self.setup_directories(f'./tb_logs_{self.architecture}', f'./checkpoints_{self.architecture}')
        self.train_model()
