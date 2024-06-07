# Uncertainty-Aware Panoptic Quality (UPQ) Evaluation Script

This script is designed to evaluate the performance of a panoptic segmentation model using the Uncertainty-Aware Panoptic Quality (AUPQ) metric. 
This metric is an extension of the Panoptic Quality (PQ) metric that takes into account the model's uncertainty in its predictions.

### Usage

```commandline
python uncertainty_aware_panoptic_quality.py --gt_json_file <gt_json_file> --pred_json_file <pred_json_file> --gt_folder <gt_folder> --pred_folder <pred_folder> --nr_thresholds <nr_thresholds>
```

- `--gt_json_file`: The path to the JSON file with ground truth data.
- `--pred_json_file`: The path to the JSON file with predictions data.
- `--gt_folder`: (optional) The folder with ground truth COCO format segmentations. Default: X if the corresponding json file is X.json.
- `--pred_folder`: (optional) The folder with prediction COCO format segmentations. Default: X if the corresponding json file is X.json.
- `--nr_thresholds`: (optional) The number of thresholds for uncertainty. Default is 16.

### Acknowledgements

This script is based on the official [COCO Panoptic Quality (PQ) evaluation script](https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py).
