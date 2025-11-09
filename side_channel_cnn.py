"""
Side-Channel Attack Detection using CNN for Agentic AI
Implements power analysis attack detection using Convolutional Neural Networks
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SCAConfig:
    """Side-Channel Attack Configuration"""
    trace_length: int = 5000
    num_classes: int = 256  # For byte-level key recovery
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2


class SideChannelCNN:
    """CNN Model for Side-Channel Attack Detection"""
    
    def __init__(self, config: SCAConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self) -> keras.Model:
        """Build CNN architecture for side-channel analysis"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv1D(64, kernel_size=11, activation='relu', 
                         input_shape=(self.config.trace_length, 1),
                         padding='same'),
            layers.BatchNormalization(),
            layers.AveragePooling1D(pool_size=2),
            
            # Second convolutional block
            layers.Conv1D(128, kernel_size=11, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.AveragePooling1D(pool_size=2),
            
            # Third convolutional block
            layers.Conv1D(256, kernel_size=11, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.AveragePooling1D(pool_size=2),
            
            # Fourth convolutional block
            layers.Conv1D(512, kernel_size=11, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.AveragePooling1D(pool_size=2),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.config.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"CNN model built with {model.count_params()} parameters")
        return model
    
    def preprocess_traces(self, traces: np.ndarray) -> np.ndarray:
        """Preprocess power traces with normalization"""
        traces_normalized = self.scaler.fit_transform(traces)
        traces_reshaped = traces_normalized.reshape(-1, self.config.trace_length, 1)
        return traces_reshaped
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Train the CNN model on power traces"""
        logger.info("Starting training...")
        start_time = time.time()
        
        # Preprocess traces
        X_train_processed = self.preprocess_traces(X_train)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_processed, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'training_time': training_time,
            'final_accuracy': self.history.history['accuracy'][-1],
            'final_val_accuracy': self.history.history['val_accuracy'][-1]
        }
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict key bytes from power traces"""
        X_test_processed = self.scaler.transform(X_test).reshape(-1, self.config.trace_length, 1)
        predictions = self.model.predict(X_test_processed)
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        X_test_processed = self.scaler.transform(X_test).reshape(-1, self.config.trace_length, 1)
        loss, accuracy = self.model.evaluate(X_test_processed, y_test, verbose=0)
        
        # Calculate key rank
        predictions = self.predict(X_test)
        key_rank = self._calculate_key_rank(predictions, y_test)
        
        results = {
            'test_loss': float(loss),
            'test_accuracy': float(accuracy),
            'average_key_rank': float(key_rank)
        }
        
        logger.info(f"Evaluation - Accuracy: {accuracy:.4f}, Key Rank: {key_rank:.2f}")
        return results
    
    def _calculate_key_rank(self, predictions: np.ndarray, true_keys: np.ndarray) -> float:
        """Calculate average key rank metric"""
        ranks = []
        for pred, true_key in zip(predictions, true_keys):
            sorted_indices = np.argsort(pred)[::-1]
            rank = np.where(sorted_indices == true_key)[0][0]
            ranks.append(rank)
        return np.mean(ranks)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class SyntheticDataGenerator:
    """Generate synthetic power traces for demonstration"""
    
    @staticmethod
    def generate_aes_traces(num_traces: int, trace_length: int, 
                           snr: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic AES power traces with Hamming weight leakage model
        
        Args:
            num_traces: Number of power traces to generate
            trace_length: Length of each trace
            snr: Signal-to-noise ratio in dB
        
        Returns:
            Tuple of (traces, labels) where labels are key bytes
        """
        logger.info(f"Generating {num_traces} synthetic traces...")
        
        # Random plaintexts and key bytes
        plaintexts = np.random.randint(0, 256, size=num_traces, dtype=np.uint8)
        key_byte = np.random.randint(0, 256, size=num_traces, dtype=np.uint8)
        
        # AES S-box (simplified for demonstration)
        sbox = np.arange(256, dtype=np.uint8)
        np.random.shuffle(sbox)
        
        traces = np.zeros((num_traces, trace_length))
        
        for i in range(num_traces):
            # Simulate AES operation
            intermediate = sbox[plaintexts[i] ^ key_byte[i]]
            
            # Hamming weight leakage model
            hw = bin(intermediate).count('1')
            
            # Generate trace with leakage at specific point
            leakage_point = trace_length // 2
            signal_power = 10 ** (snr / 10)
            noise_power = 1.0
            
            # Base noise
            trace = np.random.normal(0, np.sqrt(noise_power), trace_length)
            
            # Add leakage signal
            trace[leakage_point-50:leakage_point+50] += hw * np.sqrt(signal_power) * \
                np.exp(-0.5 * ((np.arange(100) - 50) / 10) ** 2)
            
            traces[i] = trace
        
        logger.info("Synthetic trace generation complete")
        return traces, key_byte


def main():
    """Main execution for side-channel attack detection"""
    
    # Configuration
    config = SCAConfig(
        trace_length=5000,
        num_classes=256,
        batch_size=32,
        epochs=50,
        learning_rate=0.001
    )
    
    # Generate synthetic dataset (replace with real dataset like ASCAD or DPA Contest)
    logger.info("=== Generating Synthetic Dataset ===")
    num_samples = 10000
    generator = SyntheticDataGenerator()
    traces, labels = generator.generate_aes_traces(
        num_traces=num_samples,
        trace_length=config.trace_length,
        snr=5.0
    )
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        traces, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {X_train.shape[0]} traces")
    logger.info(f"Test set: {X_test.shape[0]} traces")
    
    # Build and train model
    logger.info("\n=== Building CNN Model ===")
    sca_model = SideChannelCNN(config)
    sca_model.build_model()
    
    logger.info("\n=== Training Model ===")
    training_results = sca_model.train(X_train, y_train)
    
    # Evaluate model
    logger.info("\n=== Evaluating Model ===")
    eval_results = sca_model.evaluate(X_test, y_test)
    
    # Save results
    results = {
        'config': {
            'trace_length': config.trace_length,
            'num_classes': config.num_classes,
            'batch_size': config.batch_size,
            'epochs': config.epochs
        },
        'training': training_results,
        'evaluation': eval_results
    }
    
    with open('sca_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n=== Results Summary ===")
    logger.info(f"Training Accuracy: {training_results['final_accuracy']:.4f}")
    logger.info(f"Validation Accuracy: {training_results['final_val_accuracy']:.4f}")
    logger.info(f"Test Accuracy: {eval_results['test_accuracy']:.4f}")
    logger.info(f"Average Key Rank: {eval_results['average_key_rank']:.2f}")
    
    # Save model
    sca_model.save_model('side_channel_cnn_model.h5')
    
    logger.info("\nTraining complete! Model saved as 'side_channel_cnn_model.h5'")


if __name__ == "__main__":
    main()
