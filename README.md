# NG-Masked-RCNN

This repository contains the deployment of Masked RCNN for the 2024 Northrop Grumman Object Detection Challenge.

## Deployment

1. For general setup instructions of Masked RCNN, refer to our [main guide](https://github.com/matterport/Mask_RCNN).

2. Due to hardware limitations (specifically for training & running on Apple Silicon), we recommend deploying on an AWS EC2 instance:
   - [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/)
   - [AWS Masked RCNN Implementation](https://github.com/aws-samples/mask-rcnn-tensorflow)
   - [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
   - [Deploy Instance Segmentation Model Tutorial](https://medium.com/innovation-res/deploy-your-instance-segmentation-model-using-aws-sagemaker-part-1-e95fbeff97f1)

## Results & Findings

While this was a quick demonstration project to explore image segmentation capabilities, there are several potential improvements:

- Apply additional image properties to the training data
- Optimize model parameter tuning  
- Further explore and understand model behavior and tuning mechanisms

This project served as an excellent introduction to image segmentation models and their practical applications.
