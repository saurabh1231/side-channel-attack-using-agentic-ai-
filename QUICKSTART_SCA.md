# Quick Start Guide: Side-Channel Attack Detection

Get started with CNN-based side-channel attack detection in 5 minutes!

## Installation

```bash
# Install required packages
pip install -r requirements_sca.txt
```

## Run Complete Demo

```bash
# Run all three demonstrations
python run_sca_demo.py
```

This will:
1. Train a CNN model on synthetic power traces
2. Deploy an autonomous security agent
3. Demonstrate multi-agent collaboration

**Expected Output:**
- Training accuracy: ~85-95%
- Security alerts generated
- Analysis reports in JSON format

## Individual Components

### 1. Train CNN Model Only

```bash
python side_channel_cnn.py
```

**Output Files:**
- `side_channel_cnn_model.h5` - Trained model
- `sca_results.json` - Training metrics

### 2. Run Security Agent

```bash
python sca_agent.py
```

**Output Files:**
- `sca_agent_memory.sqlite` - Alert database
- `sca_security_report.json` - Security report

### 3. Generate Visualizations

```bash
python sca_visualizer.py
```

**Output Files:**
- `sca_training_history.png`
- `sca_threat_distribution.png`
- `sca_confidence_distribution.png`
- `sca_alerts_timeline.png`
- `sca_key_predictions.png`

## Using in Your Code

### Basic CNN Usage

```python
from side_channel_cnn import SideChannelCNN, SCAConfig
import numpy as np

# Configure model
config = SCAConfig(
    trace_length=5000,
    num_classes=256,
    batch_size=32,
    epochs=30
)

# Create and train model
model = SideChannelCNN(config)
model.build_model()

# Train with your data
traces = np.random.normal(0, 1, (10000, 5000))
labels = np.random.randint(0, 256, 10000)
model.train(traces, labels)

# Make predictions
predictions = model.predict(test_traces)
```

### Agent-Based Monitoring

```python
from sca_agent import SCAAgent
import asyncio

async def monitor_security():
    # Initialize agent
    agent = SCAAgent(agent_id="MySecurityAgent")
    
    # Train model
    await agent.train_model(num_samples=5000)
    
    # Analyze a trace
    trace = np.random.normal(0, 1, 5000)
    alert = await agent.analyze_trace(trace, "TRACE-001")
    
    print(f"Threat Level: {alert.threat_level.value}")
    print(f"Confidence: {alert.confidence:.4f}")
    
    # Get security report
    report = await agent.get_security_report()
    print(f"Total Alerts: {report['total_alerts']}")
    
    agent.shutdown()

# Run
asyncio.run(monitor_security())
```

### Load Standard Datasets

```python
from sca_dataset_loader import ASCADDataset

# Load ASCAD dataset
ascad = ASCADDataset("datasets/ASCAD_data.h5")
train_traces, train_labels = ascad.load_dataset('train', max_traces=10000)
test_traces, test_labels = ascad.load_dataset('test', max_traces=2000)

print(f"Training set: {train_traces.shape}")
print(f"Test set: {test_traces.shape}")
```

## Expected Results

### Training Performance
- **Synthetic Data**: 85-95% accuracy
- **ASCAD Dataset**: 90-98% accuracy (with proper training)
- **Training Time**: 5-15 minutes (depends on hardware)

### Agent Monitoring
- **Traces/Second**: 10-20 (depends on hardware)
- **Alert Rate**: 20-40% (on synthetic data with leakage)
- **False Positive Rate**: <5% (with proper threshold tuning)

## Troubleshooting

### Issue: TensorFlow not found
```bash
pip install tensorflow
# or for CPU-only
pip install tensorflow-cpu
```

### Issue: Out of memory
Reduce batch size in config:
```python
config = SCAConfig(batch_size=16)  # Default is 32
```

### Issue: Training too slow
- Use GPU if available
- Reduce number of epochs
- Use smaller dataset

### Issue: Low accuracy
- Increase training samples
- Adjust learning rate
- Use real dataset instead of synthetic

## Next Steps

1. **Use Real Datasets**: Download ASCAD or DPA Contest datasets
2. **Tune Hyperparameters**: Experiment with model architecture
3. **Deploy Agents**: Integrate with existing security infrastructure
4. **Visualize Results**: Use `sca_visualizer.py` for analysis
5. **Multi-Agent Setup**: Deploy multiple agents for distributed monitoring

## File Structure

```
.
├── side_channel_cnn.py       # Core CNN model
├── sca_agent.py              # Autonomous agent
├── sca_dataset_loader.py     # Dataset management
├── sca_visualizer.py         # Visualization tools
├── run_sca_demo.py           # Complete demo
├── requirements_sca.txt      # Dependencies
├── README_SCA.md             # Full documentation
└── QUICKSTART_SCA.md         # This file
```

## Support

For detailed documentation, see `README_SCA.md`

For integration with existing agents, check the main project documentation.

## Performance Tips

1. **Use GPU**: Install `tensorflow-gpu` for 10-50x speedup
2. **Batch Processing**: Process multiple traces together
3. **Model Caching**: Save and load trained models
4. **Async Operations**: Use agent's async methods for concurrency
5. **Database Indexing**: Add indexes to SQLite for faster queries

Happy detecting! 🔒🔍
