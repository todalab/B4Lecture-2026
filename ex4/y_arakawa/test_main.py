import pickle

from main import (
    calculate_forward_likelihood,
    evaluate_model_selection_by_forward_algorithm,
    evaluate_model_selection_by_viterbi_algorithm,
    forward_algorithm,
    run_forward_algorithm_against_all_output_series,
    run_viterbi_algorithm_against_all_output_series,
    viterbi_algorithm,
)


def test_forward_algorithm():
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    forward_prob = forward_algorithm(
        data1["models"]["PI"],
        data1["models"]["A"],
        data1["models"]["B"],
        data1["output"][0],
    )
    print(forward_prob)


def test_calculate_likelihood():
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    forward_prob = forward_algorithm(
        data1["models"]["PI"],
        data1["models"]["A"],
        data1["models"]["B"],
        data1["output"][0],
    )
    likelihood = calculate_forward_likelihood(forward_prob)
    print(likelihood)


def test_run_forward_algorithm_against_all_output_series():
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    likelihoods, time = run_forward_algorithm_against_all_output_series(
        data1["models"]["PI"],
        data1["models"]["A"],
        data1["models"]["B"],
        data1["output"],
    )
    print(likelihoods)
    print(f"Time: {time:.4f} seconds")


def test_evaluate_model_selection_by_forward_algorithm():
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    likelihoods, time = run_forward_algorithm_against_all_output_series(
        data1["models"]["PI"],
        data1["models"]["A"],
        data1["models"]["B"],
        data1["output"],
    )
    evaluate_model_selection_by_forward_algorithm(
        likelihoods, data1["answer_models"], time, data_number=1
    )


# def test_backward_algorithm():
#     data1 = pickle.load(open("../data/data1.pickle", "rb"))
#     backward_prob = backward_algorithm(
#         data1["models"]["PI"],
#         data1["models"]["A"],
#         data1["models"]["B"],
#         data1["output"],
#     )
#     print(backward_prob)


def test_viterbi_algorithm():
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    best_paths = viterbi_algorithm(
        data1["models"]["PI"],
        data1["models"]["A"],
        data1["models"]["B"],
        data1["output"][0],
    )
    print(best_paths)


def test_run_viterbi_algorithm_against_all_output_series():
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    best_paths, likelihoods, time = run_viterbi_algorithm_against_all_output_series(
        data1["models"]["PI"],
        data1["models"]["A"],
        data1["models"]["B"],
        data1["output"],
    )
    print(best_paths)
    print(likelihoods)
    print(f"Time: {time:.4f} seconds")


def test_evaluate_model_selection_by_viterbi_algorithm():
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    best_paths, likelihoods, time = run_viterbi_algorithm_against_all_output_series(
        data1["models"]["PI"],
        data1["models"]["A"],
        data1["models"]["B"],
        data1["output"],
    )
    evaluate_model_selection_by_viterbi_algorithm(
        best_paths, likelihoods, data1["answer_models"], time, data_number=1
    )


if __name__ == "__main__":
    # test_forward_algorithm()
    # test_calculate_likelihood()
    # test_evaluate_model_selection_by_forward_algorithm()
    # test_run_forward_algorithm_against_all_output_series()
    # test_viterbi_algorithm()
    # test_run_viterbi_algorithm_against_all_output_series()
    test_evaluate_model_selection_by_viterbi_algorithm()
