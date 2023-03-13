# Totalsegmentator Mini

Totalsegmentator Mini is a small-scale clone of [Totalsegmentator](https://github.com/wasserth/TotalSegmentator), a tool that utilizes the MONAI deep learning framework to perform semantic segmentation on medical images. This tool can be used for both inference and training tasks.

<img src="https://user-images.githubusercontent.com/37253540/216309343-ab6e3d64-2f13-43b4-93c0-4fa85e8e57fa.png"  width="300" height="300">

## Challenge

In our laboratory, we use PyTorch together with MONAI. Therefore, I would like you to become familiar with it as well. MONAI is a framework that simplifies deep learning with medical data. It achieves this by offering transforms, workflows, and networks that are tailored to the specific needs of medical image processing. Additionally, MONAI provides toolkits that enable models to be shared and deployed more easily. One such toolkit that we work with extensively is the MONAI bundle. For more information, please refer to the official MONAI documentation.

To test your ability to become familiar with the MONAI bundle, I have made some changes in the configurations files and modified  certain lines of code. As a result, the bundle is currently not working correctly. Your challenge is to fix the bundle and use it to make predictions on a single image that I will provide to you.

Please read all configuration carefully, some information that is missing/has been changed in one file might be available in another! Overall, I have added six bugs to the code. Please describe them here: 

1. 
2. 
3. 
4. 
5. 
6. 

Additionally I would like to have your input on these four questions: 

7. Your model does train, but it does not seem to learn something. How do you approach this problem? What are steps you take to narrow down the problem. 
8. Why is the Dice coefficient a better metric for segmentation tasks than accuracy. What are limitations of the Dice coefficient? What are alternatives? How would one add them the MONAI bundle?
9. What are ways to make the provided totalsegmentator model smaller and/or faster (with some acceptable decrease in accuracy). Can you provide a minimal working example for one approach?
10. What are potential applications of this model in the kidney project? How would you try to use it?

## Inference
To run inference with the repaired bundle, run the following command.

```bash
python -m monai.bundle run inference \
  --meta_file configs/metadata.yaml \
  --config_file configs/inference.yaml \
  --logging_file configs/logging.conf
```
To run inference on a single file or a directory containing multiple image files use the `--datadir` flag

