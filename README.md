# Safety Eye

## Dataset

The dataset used for this project is SafeWalkBD, obtained from Roboflow. It contains images labeled with multiple predefined categories.

Dataset link: [SafeWalkBD](https://universe.roboflow.com/safewalkbd/safewalkbd-l8jbn)

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

The model used for obstacle detection is zero-shot-detection-transformer-torch. Detailed usage instructions can be found in the [documentation](https://docs.voxel51.com/integrations/huggingface.html#zero-shot-object-detection).

## Objective

This project specifically focuses on detecting the Obstacle category from the dataset. The goal is to classify images based on the presence of obstacles:

- Class 0: No obstacle detected
- Class 1: At least one obstacle detected

## Hackathon

This project was developed for the [Voxel51 Visual AI Hackathon](https://voxel51.com/computer-vision-events/visual-ai-hackathon-march-15-2025/).
