# Side-Channel Attack Detection using CNN with Agentic AI

A comprehensive implementation of Convolutional Neural Network (CNN) based side-channel attack detection integrated with autonomous AI agents for real-time security monitoring.

## Overview

This project implements state-of-the-art deep learning techniques for detecting side-channel attacks on cryptographic implementations, with an autonomous agent framework for continuous security monitoring.

### Key Features

- **CNN-based Attack Detection**: Deep learning model for power analysis attacks
- **Agentic AI Framework**: Autonomous agents for security monitoring
- **Standard Dataset Support**: Compatible with ASCAD, DPA Contest datasets
- **Real-time Monitoring**: Continuous trace analysis with threat detection
- **Multi-Agent Collaboration**: Distributed security analysis
- **Persistent Storage**: SQLite database for alerts and analysis history

## Architecture

### Components

1. **side_channel_cnn.py**: Core CNN model implementation
   - 4-layer convolutional architecture
   - Batch normalization and dropout
   - Key byte prediction (256 classes)
   - Hamming weight leakage model

2. **sca_agent.py**: Autonomous security agent
   - Asynchronous monitoring
   - Threat level classification
   - Alert generation and storage
   - Security report generation

3. **sca_dataset_loader.py**: Dataset management
   - ASCAD dataset loader
   - DPA Contest dataset support
   - Synthetic data generation
   - AES S-box implementation

4. **run_sca_demo.py**: Complete demonstration
   - Basic CNN training
   - Agent-based monitoring
   - Multi-agent collaboration

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Install dependencies
pip install -r requirements_sca.txt

# Verify installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

## Usage

### Quick Start

Run the complete demonstration:

```bash
python run_sca_demo.py
```

This will execute three demos:
1. Basic CNN training and evaluation
2. Single agent autonomous monitoring
3. Multi-agent collaborative analysis

### Individual Components

#### 1. Train CNN Model

```bash
python side_channel_cnn.py
```

Output:
- `side_channel_cnn_model.h5`: Trained model
- `sca_results.json`: Training metrics

#### 2. Run Security Agent

```bash
python sca_agent.py
```

Output:
- `sca_agent_memory.sqlite`: Alert database
- `sca_security_report.json`: Security analysis

#### 3. Load Standard Datasets

```bash
python sca_dataset_loader.py
```

## Using Real Datasets

### ASCAD Dataset

1. Download from: https://github.com/ANSSI-FR/ASCAD
2. Place `ASCAD_data.h5` in `datasets/` folder
3. Update path in code if needed

```python
from sca_dataset_loader import ASCADDataset

ascad = ASCADDataset("datasets/ASCAD_data.h5")
traces, labels = ascad.load_dataset('train')
```

### DPA Contest Dataset

1. Visit: http://www.dpacontest.org/home/
2. Download desired version
3. Extract to `datasets/DPAContest/`

## Model Architecture

```
Input: Power traces (5000 samples)
├── Conv1D(64, kernel=11) + BatchNorm + AvgPool
├── Conv1D(128, kernel=11) + BatchNorm + AvgPool
├── Conv1D(256, kernel=11) + BatchNorm + AvgPool
├── Conv1D(512, kernel=11) + BatchNorm + AvgPool
├── Flatten
├── Dense(4096) + Dropout(0.5)
├── Dense(4096) + Dropout(0.5)
└── Dense(256, softmax) → Key byte prediction
```

## Agent Capabilities

### Threat Levels

- **SAFE**: Confidence < 0.3
- **LOW**: Confidence 0.3-0.5
- **MEDIUM**: Confidence 0.5-0.7
- **HIGH**: Confidence 0.7-0.9
- **CRITICAL**: Confidence > 0.9

### Agent Operations

```python
# Initialize agent
agent = SCAAgent(agent_id="SecurityAgent-001")

# Train model
await agent.train_model(num_samples=10000)

# Analyze single trace
alert = await agent.analyze_trace(trace, trace_id="TRACE-001")

# Monitor stream
results = await agent.monitor_stream(trace_generator, duration=60)

# Generate report
report = await agent.get_security_report()
```

## Performance Metrics

- **Accuracy**: Classification accuracy on test set
- **Key Rank**: Average rank of correct key byte
- **Inference Time**: Time per trace analysis
- **Alert Rate**: Percentage of traces triggering alerts

## Research Background

This implementation is based on research in:

1. **Deep Learning for Side-Channel Analysis**
   - CNN architectures for power analysis
   - Transfer learning approaches
   - Ensemble methods

2. **Agentic AI Systems**
   - Autonomous security monitoring
   - Multi-agent collaboration
   - Real-time threat detection

3. **Cryptographic Security**
   - AES implementation attacks
   - Power analysis countermeasures
   - Leakage assessment

## Integration with Existing Agents

This SCA system integrates with the existing agent framework:

```python
# Use with MCP protocol
from base_agent import Agent
from sca_agent import SCAAgent

# Create hybrid agent
class SecurityMonitorAgent(Agent):
    def __init__(self):
        super().__init__()
        self.sca_agent = SCAAgent()
    
    async def handle_security_request(self, traces):
        await self.sca_agent.train_model()
        results = []
        for trace in traces:
            alert = await self.sca_agent.analyze_trace(trace, f"TRACE-{i}")
            results.append(alert)
        return results
```

## Output Files

- `side_channel_cnn_model.h5`: Trained CNN model
- `sca_results.json`: Training and evaluation metrics
- `sca_security_report.json`: Security analysis report
- `sca_agent_memory.sqlite`: Alert and session database

## Troubleshooting

### TensorFlow Installation Issues

```bash
# For CPU-only version
pip install tensorflow-cpu

# For GPU support
pip install tensorflow-gpu
```

### Memory Issues

Reduce batch size or trace count:

```python
config = SCAConfig(
    batch_size=16,  # Reduce from 32
    epochs=20       # Reduce from 50
)
```

### Dataset Not Found

The system automatically generates synthetic data if real datasets are unavailable. For production use, download standard datasets.

## Future Enhancements

- [ ] Support for more datasets (ChipWhisperer, CHES CTF)
- [ ] Advanced architectures (ResNet, Attention mechanisms)
- [ ] Federated learning for distributed agents
- [ ] Real-time hardware integration
- [ ] Countermeasure effectiveness testing

## References

1. ASCAD Database: https://github.com/ANSSI-FR/ASCAD
2. DPA Contest: http://www.dpacontest.org/
3. Deep Learning SCA: Prouff et al., "Study of Deep Learning Techniques for Side-Channel Analysis"
4. CNN Architectures: Zaid et al., "Methodology for Efficient CNN Architectures in Profiling Attacks"

## License

This project is for research and educational purposes.

## Contact

For questions or collaboration, refer to the main project documentation.
