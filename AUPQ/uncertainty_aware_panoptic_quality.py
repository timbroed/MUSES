# Based on https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
import argparse
import copy
import functools
import glob
import itertools
import json
import multiprocessing
import os
import time
import traceback
from collections import defaultdict

import numpy as np
from PIL import Image

VOID = 0
OFFSET = 256 * 256 * 256
STUFF = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23)
THINGS = (24, 25, 26, 27, 28, 31, 32, 33)


class UPQStatCat():
    def __init__(self, num_t):
        self.iou_per_t = np.zeros(num_t ** 2, dtype=float)
        self.tp_per_t = np.zeros(num_t ** 2, dtype=int)
        self.fp_per_t = np.zeros(num_t ** 2, dtype=int)
        self.fn_per_t = np.zeros(num_t ** 2, dtype=int)

    def __iadd__(self, upq_stat_cat):
        self.iou_per_t += upq_stat_cat.iou_per_t
        self.tp_per_t += upq_stat_cat.tp_per_t
        self.fp_per_t += upq_stat_cat.fp_per_t
        self.fn_per_t += upq_stat_cat.fn_per_t
        return self


class UPQStat():
    def __init__(self, num_t):
        self.num_t = num_t
        self.upq_per_cat = defaultdict(functools.partial(UPQStatCat, num_t=num_t))

    def __getitem__(self, i):
        return self.upq_per_cat[i]

    def __iadd__(self, upq_stat):
        for label, upq_stat_cat in upq_stat.upq_per_cat.items():
            self.upq_per_cat[label] += upq_stat_cat
        return self

    def upq_average(self, categories, isthing):
        n_per_t = np.zeros(self.num_t ** 2, dtype=int)
        pq_per_t = np.zeros(self.num_t ** 2, dtype=float)
        sq_per_t = np.zeros(self.num_t ** 2, dtype=float)
        rq_per_t = np.zeros(self.num_t ** 2, dtype=float)
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou_per_t = self.upq_per_cat[label].iou_per_t
            tp_per_t = self.upq_per_cat[label].tp_per_t
            fp_per_t = self.upq_per_cat[label].fp_per_t
            fn_per_t = self.upq_per_cat[label].fn_per_t

            per_class_results[label] = {
                'upq': np.zeros(self.num_t ** 2, dtype=float),
                'usq': np.zeros(self.num_t ** 2, dtype=float),
                'urq': np.zeros(self.num_t ** 2, dtype=float)
            }

            for t_idx in range(self.num_t ** 2):
                iou, tp, fp, fn = iou_per_t[t_idx], tp_per_t[t_idx], fp_per_t[t_idx], fn_per_t[t_idx]
                if tp + fp + fn == 0:
                    continue
                n_per_t[t_idx] += 1
                pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
                sq_class = iou / tp if tp != 0 else 0
                rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
                per_class_results[label]['upq'][t_idx] = pq_class
                per_class_results[label]['usq'][t_idx] = sq_class
                per_class_results[label]['urq'][t_idx] = rq_class
                pq_per_t[t_idx] += pq_class
                sq_per_t[t_idx] += sq_class
                rq_per_t[t_idx] += rq_class

            per_class_results[label]['upq'] = np.mean(per_class_results[label]['upq'])
            per_class_results[label]['usq'] = np.mean(per_class_results[label]['usq'])
            per_class_results[label]['urq'] = np.mean(per_class_results[label]['urq'])

        pq_per_t /= n_per_t
        sq_per_t /= n_per_t
        rq_per_t /= n_per_t

        avg_pq = np.mean(pq_per_t)
        avg_sq = np.mean(sq_per_t)
        avg_rq = np.mean(rq_per_t)

        assert len(np.unique(n_per_t)) == 1  # should be the same for all thresholds
        n = n_per_t[0]

        return {'aupq': avg_pq, 'ausq': avg_sq, 'aurq': avg_rq, 'n': n}, per_class_results


def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e

    return wrapper


def rgb2id(color):
    color = color.astype(np.uint32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def upq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, nr_thresholds):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(upq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories, nr_thresholds))
        processes.append(p)
    upq_stat = UPQStat(nr_thresholds)
    for p in processes:
        upq_stat += p.get()
    return upq_stat


@get_traceback
def upq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories, nr_thresholds):
    upq_stat = UPQStat(nr_thresholds)

    for idx, (gt_ann, pred_ann) in enumerate(annotation_set):
        print('Core: {}, {} from {} images processed with {}x{} grid'.format(proc_id, idx, len(annotation_set),
                                                                             nr_thresholds, nr_thresholds))

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels = np.unique(pan_pred)
        for label in labels:
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError(
                    'In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(
                        gt_ann['image_id'], label))
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError(
                    'In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'],
                                                                                                    label,
                                                                                                    pred_segms[label][
                                                                                                        'category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not '
                           'presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # load unconfidence maps
        gt_uncertainty_folder = gt_folder.replace('gt_panoptic', 'gt_uncertainty')
        unconf_gt = np.array(Image.open(
            os.path.join(gt_uncertainty_folder, gt_ann['file_name'].replace('_panoptic.png', '_uncertainty.png'))),
                             dtype=np.uint8)
        unconf_class_gt = unconf_gt == 255
        unconf_instance_gt = unconf_gt == 127

        conf_class_pred, conf_instance_pred = load_uncertainty_annotation(pred_ann, pred_folder)

        thresholds = np.linspace(0, 255, nr_thresholds)
        for t_idx, (class_t, instance_t) in enumerate(itertools.product(thresholds, repeat=2)):

            unconf_class_pred = conf_class_pred < class_t
            unconf_instance_pred = conf_instance_pred < instance_t

            # we will modify this
            this_pan_pred = np.copy(pan_pred)
            this_pred_segms = copy.deepcopy(pred_segms)

            oracle_instances = defaultdict(int)

            # Let's treat this hierarchical: if we predict unconfident_class, this implies
            # that we are not confident about the instance either.
            # So the pixels with a TP unconfident_class prediction get an oracle instance.

            # class-level
            tp_replace_class_mask = unconf_class_pred & unconf_class_gt
            fp_replace_class_mask = unconf_class_pred & ~unconf_class_gt
            # set false unconfident_class predictions to void, this means they will be discarded entirely (also on the instance-level)
            this_pan_pred[fp_replace_class_mask] = VOID
            if np.any(tp_replace_class_mask):
                category_id_to_pred_id_map = {v['category_id']: k for k, v in this_pred_segms.items() if
                                              v['category_id'] in STUFF}
                for ct in np.unique(pan_gt[tp_replace_class_mask]):

                    if ct == VOID:
                        continue

                    cat = gt_segms[ct]['category_id']
                    if cat in STUFF:
                        # replace class prediction with GT
                        cls_mask = tp_replace_class_mask & (pan_gt == ct)
                        if cat in category_id_to_pred_id_map:
                            this_pan_pred[cls_mask] = category_id_to_pred_id_map[cat]
                        else:
                            # add the new class to the prediction
                            new_pred_id = max(this_pred_segms.keys()) + 1
                            category_id_to_pred_id_map[cat] = new_pred_id
                            this_pan_pred[cls_mask] = new_pred_id
                            this_pred_segms[new_pred_id] = {'category_id': cat}

                    elif cat in THINGS:
                        # create an oracle instance
                        cls_mask = tp_replace_class_mask & (pan_gt == ct)
                        this_pan_pred[cls_mask] = VOID
                        # don't add instance if it is crowd
                        if gt_segms[ct]['iscrowd'] == 0:
                            oracle_instances[ct] += np.count_nonzero(cls_mask)

                    else:
                        print('unknown value {} in ground truth'.format(ct))
                        raise ValueError

            # instance-level
            tp_replace_instance_mask = unconf_instance_pred & (
                        unconf_instance_gt | unconf_class_gt) & ~unconf_class_pred
            fp_replace_instance_mask = unconf_instance_pred & ~(
                        unconf_instance_gt | unconf_class_gt) & ~unconf_class_pred
            # set false unconfident_instance predictions to void
            this_pan_pred[fp_replace_instance_mask] = VOID
            if np.any(tp_replace_instance_mask):
                category_pred = np.zeros_like(this_pan_pred)
                for key in this_pred_segms.keys():
                    category_id = this_pred_segms[key]['category_id']
                    category_pred[this_pan_pred == key] = category_id

                for ct in np.unique(pan_gt[tp_replace_instance_mask]):

                    if ct == VOID:
                        continue

                    cat = gt_segms[ct]['category_id']
                    if cat in STUFF:
                        # there is no effect for stuff classes
                        # this can happen when in the GT the class is uncertain (level 2) and the model predicts instance uncertain
                        # I think in this case, we should not do anything
                        pass

                    elif cat in THINGS:
                        # create an oracle instance if predicted class is right
                        cls_mask = tp_replace_instance_mask & (pan_gt == ct) & (category_pred == cat)
                        if np.any(cls_mask):
                            this_pan_pred[cls_mask] = VOID
                            # don't add instance if it is crowd
                            if gt_segms[ct]['iscrowd'] == 0:
                                oracle_instances[ct] += np.count_nonzero(cls_mask)

                    else:
                        print('unknown value {} in ground truth'.format(ct))
                        raise ValueError

            # predicted segments area calculation
            labels, labels_cnt = np.unique(this_pan_pred, return_counts=True)
            new_pred_segms = {}
            for label, label_cnt in zip(labels, labels_cnt):
                if label == VOID:
                    continue
                curr_pred_info = this_pred_segms[label]
                curr_pred_info['area'] = label_cnt
                new_pred_segms[label] = curr_pred_info
            this_pred_segms = new_pred_segms

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + this_pan_pred.astype(np.uint64)
            gt_pred_map = {}
            filtered_gt_ids = []
            filtered_pred_ids = []
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

                # filtered
                if gt_id not in gt_segms:
                    continue
                if pred_id not in this_pred_segms:
                    continue
                if gt_segms[gt_id]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_id]['category_id'] != this_pred_segms[pred_id]['category_id']:
                    continue
                if pred_id == VOID:
                    continue
                filtered_gt_ids.append(gt_id)
                filtered_pred_ids.append(pred_id)

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            for gt_label, pred_label in zip(filtered_gt_ids, filtered_pred_ids):
                matching_intersection = gt_pred_map[(gt_label, pred_label)]
                matching_union = this_pred_segms[pred_label]['area'] + gt_segms[gt_label][
                    'area'] - matching_intersection - gt_pred_map.get((VOID, pred_label), 0) - oracle_instances.get(
                    gt_label, 0)
                matching_iou = matching_intersection / matching_union
                if matching_iou > 0.5:
                    # iou >= matching_iou
                    intersection = matching_intersection + oracle_instances.get(gt_label, 0)
                    union = this_pred_segms[pred_label]['area'] + gt_segms[gt_label][
                        'area'] - matching_intersection - gt_pred_map.get((VOID, pred_label), 0)
                    iou = intersection / union
                    upq_stat[gt_segms[gt_label]['category_id']].tp_per_t[t_idx] += 1
                    upq_stat[gt_segms[gt_label]['category_id']].iou_per_t[t_idx] += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    oracle_instances.pop(gt_label, None)

            # for the remaining oracles, check if by themselves they match
            for gt_label, intersection in oracle_instances.items():
                union = gt_segms[gt_label]['area']
                iou = intersection / union
                if iou > 0.5:
                    upq_stat[gt_segms[gt_label]['category_id']].tp_per_t[t_idx] += 1
                    upq_stat[gt_segms[gt_label]['category_id']].iou_per_t[t_idx] += iou
                    gt_matched.add(gt_label)

            # count false negatives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                upq_stat[gt_info['category_id']].fn_per_t[t_idx] += 1

            # count false positives
            for pred_label, pred_info in this_pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                upq_stat[pred_info['category_id']].fp_per_t[t_idx] += 1

    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return upq_stat


def load_uncertainty_annotation(pred_ann, pred_folder):
    class_confidence_folder = pred_folder.replace('labelIds', 'classConfidence')
    instance_confidence_folder = pred_folder.replace('labelIds', 'instanceConfidence')

    class_confidence_pattern = pred_ann['file_name'].replace('.png', '*.png')
    instance_confidence_pattern = pred_ann['file_name'].replace('.png', '*.png')

    class_confidence_files = glob.glob(os.path.join(class_confidence_folder, class_confidence_pattern))
    instance_confidence_files = glob.glob(os.path.join(instance_confidence_folder, instance_confidence_pattern))

    # Check if more than one file matches each pattern
    if len(class_confidence_files) > 1:
        raise Exception("Multiple class confidence files present in classConfidence/.")
    elif len(class_confidence_files) == 0:
        raise Exception("No class confidence files match the pattern.")
    if len(instance_confidence_files) > 1:
        raise Exception("Multiple instance confidence files present in instanceConfidence/.")
    elif len(class_confidence_files) == 0:
        raise Exception("No instance confidence files match the pattern.")

    # Load the first matching file for each pattern if available
    conf_class_pred = np.array(Image.open(class_confidence_files[0]), dtype=np.uint8)
    conf_instance_pred = np.array(Image.open(instance_confidence_files[0]), dtype=np.uint8)

    return conf_class_pred, conf_instance_pred


def upq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None, nr_thresholds=16):
    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))
    print("\tNumber thresholds: {}".format(nr_thresholds))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    upq_stat = upq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, nr_thresholds)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = upq_stat.upq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "AUPQ", "AUSQ", "AURQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['aupq'],
            100 * results[name]['ausq'],
            100 * results[name]['aurq'],
            results[name]['n'])
        )

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str, required=True,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str, required=True,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground truth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--nr_thresholds', type=int, default=16,
                        help="Number of thresholds for uncertainty. Default is 16.")

    args = parser.parse_args()
    upq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder, args.nr_thresholds)