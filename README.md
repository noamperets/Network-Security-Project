# 🌐 Network Simulator

This project is a network simulator written in Python. It simulates a network of nodes and links, and uses various algorithms to analyze the network's performance under different conditions.

## ✨ Features

- 🧩 Generation of random network topologies with a specified number of nodes and links
- 📊 Calculation of network parameters such as gain, interference, and capacity
- 🚀 Simulation of data flows through the network
- 🗺️ Visualization of the network graph
- 📉 Analysis of network performance under different conditions

## 📁 Code Structure

The project consists of several Python files, with the main one being `network_simulator.py`:

This file defines the main classes used in the simulation, including:
- `Node`
- `Link`
- `Route`
- `Simulator`

It also includes functions for:
- generating the network graph
- calculating network parameters
- visualizing the network

There are also additional Python files in the project that handle various aspects of the network simulation. However, `network_simulator.py` is the core file where the main simulation classes and functions are defined.

## ▶️ Usage

To run the simulation, execute the `main` function in the selected file:
```bash
python network_simulator.py
```
You can customize the simulation by modifying the parameters in the initialize_sim function in network_simulator.py and also replace the default functions for calculating network parameters with your own.
