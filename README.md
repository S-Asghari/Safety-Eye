# Safety Eye

## Dataset

The dataset used for this project is SafeWalkBD, obtained from Roboflow. It contains images labeled with multiple predefined categories.

Dataset link: SafeWalkBD

## Predefined Categories

The original dataset includes the following 16 categories:
- Animal
- Crosswalk
- Obstacle
- Over-bridge
- Person
- Pole
- Pothole
- Railway
- Road-barrier
- Sidewalk
- Stairs
- Traffic-light
- Traffic-sign
- Train
- Tree
- Vehicle

## Prediction Model

The model used for obstacle detection is zero-shot-detection-transformer-torch. Detailed usage instructions can be found in the documentation:

Model reference: Zero-Shot Object Detection

## Objective

This project specifically focuses on detecting the Obstacle category from the dataset. The goal is to classify images based on the presence of obstacles:

- Class 0: No obstacle detected
- Class 1: At least one obstacle detected
