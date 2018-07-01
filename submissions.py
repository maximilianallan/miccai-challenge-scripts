import os
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import ndimage
from skimage import morphology

plt.style.use('ggplot')

class Label:

    def __init__(self, name, val):

        self.name = name
        self.val = val


class Submission:

    def __init__(self, name, institute, method, path, root_data_dir):

        self.dir = os.path.join(root_data_dir, "submissions/instrument/")

        assert(os.path.exists(self.dir))

        self.dataset_results = []
        self.name = name
        self.institute = institute
        self.method = method

    def get_number_of_datasets(self):
        return len(self.dataset_results)

    def get_number_of_images_for_dataset(self, dataset_idx):
        return self.dataset_results[dataset_idx].size

    def get_short_name(self):
        return self.name.split(" ")[0].lower()[0] + self.name.split(" ")[1].lower()[0:4]

    def get_label_name(self):
        return self.name.split(" ")[0][0] + " " + self.name.split(" ")[1]

    def compute_scores_across_datasets(self):

        self.mean_iou_per_dataset = []
        if self.dataset_results[0].is_multiclass_results():
            self.total_confusion_matrix = np.zeros(
                shape=self.dataset_results[0].confusion_matrix.shape, dtype=np.float32)
            self.confusion_matrix_per_dataset = []

        for dset in self.dataset_results:
            results = [fname_im[1]
                       for fname_im in dset.results if fname_im[1] is not None]
            if len(results) > 0:
                self.mean_iou_per_dataset.append(np.mean(results))
                if dset.is_multiclass_results():
                    self.confusion_matrix_per_dataset.append(
                        dset.confusion_matrix)
                    self.total_confusion_matrix = self.total_confusion_matrix + dset.confusion_matrix
            else:
                self.mean_iou_per_dataset.append(np.inf)

        total_frames = sum([len(f.results) for f in self.dataset_results])
        if total_frames == 0:
            print("[WARNING] No submissions for {} in any dataset".format(self.name))
            self.mean_iou_all_datasets = None
            return False

        dset_lens = [float(len(f.results)) /
                     total_frames for f in self.dataset_results]

        # if they have any nans in the mean score, they don't get a score for this problem
        if np.inf in self.mean_iou_per_dataset or np.nan in self.mean_iou_per_dataset:
            print("[WARNING] nan or inf found in results for {}".format(self.name))
            self.mean_iou_all_datasets = None
        else:
            self.mean_iou_all_datasets = np.average(
                self.mean_iou_per_dataset, weights=dset_lens)  # np.mean(self.mean_iou_per_dataset)
        return self.mean_iou_all_datasets is not None

    def get_dataset_results(self, dataset_idx):
        return self.dataset_results[dataset_idx]


class ImageResults:

    def __init__(self, ground_truth_image, submission_image, error_function):

        self.error = error_function(submission_image, ground_truth_image)
        self.num_pixels = ground_truth_image.shape[0] * \
            ground_truth_image.shape[1]


class DatasetResults:

    def __init__(self, dataset_name, dataset_size, error_function):

        self.name = dataset_name
        self.size = dataset_size
        self.results = []
        self.error_function = error_function
        if self.is_multiclass_results():
            self.confusion_matrix = np.zeros(shape=(len(self.error_function.vals), len(
                self.error_function.vals)), dtype=np.float32)

    def is_multiclass_results(self):
        return isinstance(self.error_function, MulticlassIntersectionOverUnionError)

    def add_results_for_frame(self, file_name, ground_truth_image, submission_image):

        if ground_truth_image is None:
            raise Exception("[ERROR] Ground truth image should never be null!")
        else:
            error = self.error_function.get_error(
                ground_truth_image, submission_image)
            if error is not None:
                # we return None is there are no pixels for the label in the image (ground truth)
                self.results.append((file_name, error))
                if self.is_multiclass_results():
                    self.confusion_matrix = self.confusion_matrix + \
                        self.error_function.get_confusion(
                            ground_truth_image, submission_image)
                return True
            return False

    def get_per_frame_iou(self):
        return [fname_im[1] for fname_im in self.results]

    def get_frame_numbers(self):
        return [int(re.findall(r'\d+', x[0])[0]) for x in self.results]

    def is_good(self):
        return np.sum([f[1] for f in self.results]) > 0


def get_intersection_over_union(overlap, prediction_not_overlap, ground_truth_not_overlap):

    if (ground_truth_not_overlap + overlap) == 0:
        # this occurs when there is no pixels for ground truth class in image - no score in this image
        return None

    return float(overlap) / (prediction_not_overlap + ground_truth_not_overlap + overlap)


class MulticlassIntersectionOverUnionError:

    vals = []
    orig_vals = []

    def __init__(self, vals):
        MulticlassIntersectionOverUnionError.vals = vals
        MulticlassIntersectionOverUnionError.orig_vals = vals

    @staticmethod
    def reset_orig_vals():
        MulticlassIntersectionOverUnionError.vals = MulticlassIntersectionOverUnionError.orig_vals

    @staticmethod
    def switch_on_val(val):
        MulticlassIntersectionOverUnionError.vals = [val]

    def get_error(self, ground_truth_frame, entry_frame):

        ious = []

        for val in MulticlassIntersectionOverUnionError.vals:

            true_positive_count = np.sum(
                (ground_truth_frame == val.val) & (entry_frame == val.val))
            true_negative_count = np.sum(
                (ground_truth_frame != val.val) & (entry_frame != val.val))
            false_positive_count = np.sum(
                (ground_truth_frame != val.val) & (entry_frame == val.val))
            false_negative_count = np.sum(
                (ground_truth_frame == val.val) & (entry_frame != val.val))

            iou = get_intersection_over_union(
                true_positive_count, false_positive_count, false_negative_count)
            if iou is None:
                # only count IoU for mean for classes actually in this frame
                continue
            else:
                if np.isnan(iou):
                    print "Warning: found nan: tp = {}, tn = {}, fp = {}, fn = {}".format(
                        true_positive_count, true_negative_count, false_positive_count, false_negative_count)
                ious.append(iou)

        if len(ious) == 0:
            return None

        return np.mean(ious)

    def get_confusion(self, ground_truth_frame, entry_frame):
        conf = np.zeros(shape=(len(MulticlassIntersectionOverUnionError.vals), len(
            MulticlassIntersectionOverUnionError.vals)), dtype=np.float32)

        for n, is_val in enumerate(MulticlassIntersectionOverUnionError.vals):
            for m, pred_val in enumerate(MulticlassIntersectionOverUnionError.vals):

                count = np.sum((ground_truth_frame == is_val.val)
                               & (entry_frame == pred_val.val))
                conf[n, m] = count

        return conf


class IntersectionOverUnionError:

    def __init__(self):
        pass

    def get_error(self, ground_truth_frame, entry_frame):

        true_negative_count = np.sum(
            (ground_truth_frame == 0) & (entry_frame == 0))
        true_positive_count = np.sum(
            (ground_truth_frame == 255) & (entry_frame == 255))
        false_positive_count = np.sum(
            (ground_truth_frame == 0) & (entry_frame == 255))
        false_negative_count = np.sum(
            (ground_truth_frame == 255) & (entry_frame == 0))

        iou_foreground = get_intersection_over_union(
            true_positive_count, false_positive_count, false_negative_count)
        if iou_foreground is None:
            iou_foreground = 0

        return iou_foreground


class SDFError:

    def __init__(self):
        pass

    def get_error(self, ground_truth_frame, entry_frame):

        # skeletonize both
        skeleton_ground_truth = morphology.skeletonize(
            (ground_truth_frame == 255) * 1) * 1
        # entry may be too messy to effectively skeletonize...
        # morphology.skeletonize(entry_frame)
        skeleton_entry = (entry_frame == 255)*1
        # reduce to a set of contour points

        contour_ground_truth = skeleton_ground_truth == 1
        contour_entry = skeleton_entry == 1

        # generate sdf from contours
        ground_truth_distance_transform = ndimage.distance_transform_edt(
            np.invert(skeleton_ground_truth))
        entry_frame_distance_transform = ndimage.distance_transform_edt(
            np.invert(skeleton_entry))
        # sum the sdf indexed by the contour for each and take average

        if np.sum(skeleton_entry) > 0:
            precision = np.sum(
                entry_frame_distance_transform[contour_ground_truth]) / np.sum(skeleton_entry)
        else:
            precision = None

        if np.sum(skeleton_ground_truth) > 0:
            recall = np.sum(
                ground_truth_distance_transform[contour_entry]) / np.sum(skeleton_ground_truth)
        else:
            recall = None

        if precision is None and recall is None:
            return 0
        elif precision is None:
            return 3.0*recall/2
        elif recall is None:
            return 3.0*precision/2

        return (precision + recall)/2
