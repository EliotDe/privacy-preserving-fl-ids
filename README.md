# FLIDS-Tradeoff: Federated Learning for Privacy-Preserving Intrusion Detection in IoT Networks

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flower%20(flwr)-orange.svg)](https://flower.ai/)

This repository contains the implementation, experimental framework, and core findings for an AI-driven, privacy-preserving Intrusion Detection System (IDS) designed for IoT networks using Federated Learning (FL). 

A comprehensive, 20-page research report detailing the literature review, deep experimental methodology, and implementation design is available in the root directory: `final_report-5-1.pdf`.

---

## Project Overview

Traditional centralized IoT IDS models violate privacy and put a strain on their networks by transmitting network traffic data to a central server for analysis. This project simulates a decentralized approach using **Flower** to train a **1D-CNN network traffic classifier** across heterogeneous clients without exposing raw data. FL-based IDSs are vulnerable to gradient inversion attacks (i.e. deep leakage style attacks) which recover training data from model updates; threat actors can use this recovered data to time attacks so that they appear like normal traffic. This project provides an in-depth look at how effective differential privacy is as a defense to these inversion attacks and how it affects model bias and scalability.  

### Key Contributions & Features:
* **Realistic Non-I.I.D. Data Partitioning:** Built a custom preprocessing pipeline to simulate temporally realistic, heterogeneous network traffic distributions across clients.
* **Empirical Privacy Auditing:** Implemented three distinct inversion attacks to rigorously test and measure privacy leakage.
* **Privacy-Utility Optimization:** Evaluated the scalability, bias, and accuracy trade-offs of introducing Local Differential Privacy (LDP).

---

## Experimental Results & Key Findings

*Detailed charts, evaluation metrics, and analysis can be found in the attached report.*

## Repository Structure

```text
├── flids/                  # Core FL-based intrusion detection simulation code
├── pyproject.toml          # Project configuration file
├── final_report-5-1.pdf    # Full 20-page research paper & methodology
└── requirements.txt        # Project dependencies
...                         # Experiment Scripts
```

## Running the Simulation

My project used the ToN_IoT network traffic datasets with a custom preprocessing pipeline so the processed dataset I used isn't actually available; I haven't published it yet, so unfortuanately you can't run this project at the moment from the cloned repo. If you still want to mess around with everything or tweak the project for your own IoT network traffic here are the installation instructions.

## Setup and Installation

### Prerequisite: OS Compatibility Note

**Windows Users:** The Flower framework utilizes the ray module for client simulations, which can experience performance and threading issues natively on Windows. Flower recommends running via WSL.

### 1. Clone & Environment Setup

```
git clone [https://github.com/EliotDe/FLIDS-Tradeoff.git](https://github.com/EliotDe/FLIDS-Tradeoff.git)
cd FLIDS-Tradeoff
```

Create and activate a virtual environment:

```
# On Linux/WSL
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Configuration Setup

Flower requires a specific global configuration file for tracking metrics and simulation parameters. Copy the provided configuration file to your home .flwr/ directory:

```
# Ensure the directory exists, then copy the config
mkdir -p ~/.flwr/
cp config.toml ~/.flwr/
```


