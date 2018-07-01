from skimage.io import imread
import os
from submissions import *
import json
import matplotlib.pyplot as plt
import re
import sys

# Return line plot markers for each submission


def get_color_and_marker(idx):

    if idx == 0:
        return ("b", "o")
    elif idx == 1:
        return ("g", "o")
    elif idx == 2:
        return ("r", "o")
    elif idx == 3:
        return ("c", "o")
    elif idx == 4:
        return ("m", "o")
    elif idx == 5:
        return ("y", "o")
    elif idx == 6:
        return ("b", "x")
    elif idx == 7:
        return ("g", "x")
    elif idx == 8:
        return ("r", "x")
    elif idx == 9:
        return ("c", "x")
    elif idx == 10:
        return ("m", "x")
    elif idx == 11:
        return ("y", "x")
    else:
        assert(0)

# Returns the error function (mean IoU for multiclass and binary) and the labels
# for each problem


def get_error_and_labels(results_type, raw_data_dir):

    if results_type == "BinarySegmentation":

        error_function = IntersectionOverUnionError()
        labels = [Label("Foreground", 255), Label("Background", 0)]

    elif results_type == "PartsSegmentation":

        with open(os.path.join(raw_data_dir, 'instrument_class_mapping.json')) as data_file:
            data = json.load(data_file)
            labels = [Label(d, data[d]) for d in data]
        error_function = MulticlassIntersectionOverUnionError(labels)

    elif results_type == "TypeSegmentation":

        with open(os.path.join(raw_data_dir, 'instrument_type_mapping.json')) as data_file:
            data = json.load(data_file)
            labels = [Label(d, data[d]) for d in data]
        error_function = MulticlassIntersectionOverUnionError(labels)

    return (error_function, labels)

# Process the submissions for each different challenge
# @param results_type The 'type' of the results from BinarySegmentation/PartsSegmentation/TypeSegmentation
# @param error_function The error function we use to process. From get_error_and_labels().


def process(results_type, error_function, data_dir):

    subs = [Submission("Seb Bodenstedt", "NCT", "Method 7", "SebBodenstedt", data_dir),
            Submission("Thomas Kurmann", "UB", "Method 8",
                       "ThomasKurmann", data_dir),
            Submission("Shiyun Zhou", "BIT", "Method 1",
                       "BIToptics", data_dir),
            Submission("Vladimir Iglovikov", "MIT", "Method 3",
                       "IglovikovShevts", data_dir),
            Submission("Houling Luo", "SIAT", "Method 2",
                       "HoulingLuo", data_dir),
            Submission("Luis Herrera", "UCL", "Method 4",
                       "LuisHerrera", data_dir),
            Submission("Nicola Reike", "TUM", "Method 5",
                       "NicolaReike", data_dir),
            Submission("Rahul Duggal", "Delhi", "Method 6",
                       "RahulDuggal", data_dir),
            Submission("Vincent Zhang", "UA", "Method 9",
                       "VincentZhang", data_dir),
            Submission("Yun Hsuan Su", "UW", "Method 10", "YunHsuanSu", data_dir)]

    NUM_GT_DIRS = 10
    ground_truth_dirs = ["instrument_dataset_{0}".format(
        i) for i in range(1, NUM_GT_DIRS+1)]

    # Iterate over all of the datasets
    # Load the ground truth frames for the dataset incrementally
    # For each submission, compute the results for this frame and add
    # then to a list of scores
    os.chdir(os.path.join(args.data, "processed_labels/instrument/"))
    cwd = os.getcwd()
    for dataset_dir in ground_truth_dirs:

        os.chdir(dataset_dir)
        print("[STATUS] Processing " + dataset_dir)

        for sub in subs:

            sub.dataset_results.append(DatasetResults(
                dataset_dir, len(os.listdir(".")), error_function))

        for frame in sorted(os.listdir(results_type), key=lambda x: re.findall(r'\d+', x)):

            ground_truth_image = imread(
                results_type + "/" + frame, as_gray=True)

            # check that if we skip the frame due to no ground truth pixels that we do this for all frames
            for sub in subs:
                submission_image = imread(os.path.join(
                    sub.dir, results_type, dataset_dir, frame), as_gray=True)
                if submission_image is None:
                    print("[WARNING] Image is not found : " +
                          os.path.join(sub.dir, results_type, dataset_dir, frame))
                    continue
                sub.dataset_results[-1].add_results_for_frame(
                    frame, ground_truth_image, submission_image)

        os.chdir(cwd)

    return subs


def generate_per_dataset_line(dataset_idx, submissions, title, results_dir):

    x_axis = np.array(submissions[0].get_dataset_results(
        dataset_idx).get_frame_numbers())

    plt.rcParams["figure.figsize"] = (50, 20)
    plt.rcParams.update({'font.size': 22})

    fig = plt.figure()

    for idx, submission in enumerate(submissions):
        line_color, line_marker = get_color_and_marker(idx)
        if not submission.get_dataset_results(dataset_idx).is_good():
            continue
        ious_with_Nones = submission.get_dataset_results(
            dataset_idx).get_per_frame_iou()
        ious_with_Nones = np.array(ious_with_Nones).astype(np.double)
        good_vals = np.isfinite(ious_with_Nones)
        plt.plot(x_axis[good_vals], ious_with_Nones[good_vals], label=submission.get_label_name(
        ), linestyle='-', marker=line_marker, color=line_color)

    ax = plt.gca()
    ax.set_title("{0} - Dataset {1}".format(title, dataset_idx+1))
    ax.set_xlabel('Frame No.')
    ax.set_ylabel('IOU')

    plt.legend()

    save_dir = os.path.join(results_dir, "dataset_" + str(dataset_idx+1))

    try:
        os.makedirs(save_dir)
    except:
        pass

    fig.savefig(save_dir + "/" + title.replace(" ", "_") + ".png")
    plt.close(fig)


def generate_plots(challenge, submissions, title, base_dir):

    print("[STATUS] Done loading. Generating plots")
    results_dir = os.path.join(
        base_dir, "results_FIXED/instrument/", title.replace(" ", "_"))
    try:
        os.makedirs(results_dir)
    except:
        pass

    # work out which entrants have submissions
    submissions_with_entries = []
    for submission in submissions:
        if submission.compute_scores_across_datasets():
            submissions_with_entries.append(submission)
        else:
            print("[WARNING] Excluding submission {} from challenge {}".format(
                submission.name, challenge))

    if len(submissions_with_entries) == 0:
        print("[WARNING] No submission with entries found for challenge {}".format(
            challenge))
        return

    with open(os.path.join(results_dir, "total_scores.txt"), "w") as f:

        for submission in submissions_with_entries:
            f.write("{0} : {1}\n".format(submission.name,
                                         submission.mean_iou_all_datasets))

        #f.write("\nConfusion Matrices\n")
        # for submission in submissions:
        #    f.write("{0} : \n{1}\n".format(submission.name, submission.total_confusion_matrix))

    with open(os.path.join(results_dir, "latex_table.tex"), "w") as l:

        l.write("\\begin{{table*}}\n\\centering\n\\begin{{tabular}}{{ l | {} c }}\n".format(
            "".join([" c |"] * (len(submissions)-1))))
        for submission in submissions_with_entries:
            l.write(" & {}".format(submission.institute))
        l.write(" \\\\\n\hline\n")

        for dset in range(submissions_with_entries[0].get_number_of_datasets()):
            l.write("Dataset {} ".format(dset))
            generate_per_dataset_line(
                dset, submissions_with_entries, title, results_dir)
            with open(os.path.join(results_dir, "dataset_" + str(dset+1), "scores.txt"), "w") as f:
                for submission in submissions_with_entries:
                    f.write("{0} : {1}\n".format(submission.name,
                                                 submission.mean_iou_per_dataset[dset]))
                    l.write(" & {0:.3f} ".format(
                        submission.mean_iou_per_dataset[dset]))
            l.write("\\\\\n")
            #f.write("\nConfusion Matrices\n")
            # for submission in submissions:
            #    f.write("{0} : \n{1}\n".format(submission.name, submission.confusion_matrix_per_dataset[dset]))
        l.write("\\end{tabular}\n")
        l.write(
            "\\caption{{\label{{fig:{}}} A caption...}}\n\\end{{table*}}\n".format(challenge))


def get_title_from_challenge(challenge):
    if challenge == "BinarySegmentation":
        return "Binary Segmentation"
    elif challenge == "PartsSegmentation":
        return "Parts-based Segmentation"
    elif challenge == "TypeSegmentation":
        return "Type Segmentation"
    else:
        assert(0)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Process MICCAI 2017 results.')
    parser.add_argument(
        'data', type=str, help='The directory where the data is stored.')

    args = parser.parse_args()

    raw_data_dir = os.path.join(args.data, "training_data/instrument/")

    if not os.path.exists(raw_data_dir):
        print("[ERROR] Unable to find data directories in {}".format(raw_data_dir))
        sys.exit(1)

    # Compute the binary segmentation results
    challenge = "BinarySegmentation"
    error_and_labels = get_error_and_labels(challenge, raw_data_dir)
    submissions = process(challenge, error_and_labels[0], args.data)
    generate_plots(challenge, submissions,
                   get_title_from_challenge(challenge), args.data)

    # Compute the parts-based segmentation results
    challenge = "PartsSegmentation"
    error_and_labels = get_error_and_labels(challenge, raw_data_dir)
    submissions = process(challenge, error_and_labels[0], args.data)
    generate_plots(challenge, submissions,
                   get_title_from_challenge(challenge), args.data)

    # Compute one-against-all results for each part
    for label in error_and_labels[1]:
        MulticlassIntersectionOverUnionError.switch_on_val(label)
        submissions = process(challenge, error_and_labels[0], args.data)
        generate_plots(challenge, submissions,
                       "{0} Part Segmentation".format(label.name), args.data)
        MulticlassIntersectionOverUnionError.reset_orig_vals()

    # Compute the type-based segmentation results
    challenge = "TypeSegmentation"
    error_and_labels = get_error_and_labels(challenge, raw_data_dir)
    submissions = process(challenge, error_and_labels[0], args.data)
    generate_plots(challenge, submissions,
                   get_title_from_challenge(challenge), args.data)

    # Compute one-against-all results for each type
    for label in error_and_labels[1]:
        MulticlassIntersectionOverUnionError.switch_on_val(label)
        submissions = process(challenge, error_and_labels[0], args.data)
        generate_plots(challenge, submissions,
                       "{0} Type Segmentation".format(label.name), args.data)
        MulticlassIntersectionOverUnionError.reset_orig_vals()
